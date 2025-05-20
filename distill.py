import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.llama.modeling_llama import LlamaMLP
from datasets import load_dataset
from tqdm.auto import tqdm
import json, ast
from datetime import datetime

# Configuration
TEACHER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
STUDENT_INTERMEDIATE_SIZE = 2048  # Reduced from 8192
DATASET_NAME = "glaiveai/glaive-function-calling-v2"
OUTPUT_DIR = "./distilled_llama_mlp"

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 1024
MAX_SAMPLES = 2000
EVAL_SAMPLES = 200

# Distillation hyperparameters
TEMPERATURE = 2.0
ALPHA_CE = 0.4  # Weight for Cross-Entropy loss
ALPHA_KD = 0.6  # Weight for KL Divergence loss

# Device setup
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# Load teacher model
print("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device)
teacher_model.eval()

# Create student model by first loading it as a copy of teacher model
print("Creating student model...")
student_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device)

# Modify MLP intermediate size and reinitialize MLP weights
print("Modifying MLP size and reinitializing weights...")
for name, module in student_model.named_modules():
    if "mlp" in name:
        # Get the original intermediate size
        student_config = teacher_model.config
        student_config.intermediate_size = STUDENT_INTERMEDIATE_SIZE
        module = LlamaMLP(student_config)
        # print(module)

        # original_size = module.intermediate_size
        # # Modify intermediate size
        # module.intermediate_size = STUDENT_INTERMEDIATE_SIZE

        # # Reinitialize the MLP weights with new dimensions
        # if hasattr(module, 'up_proj'):
        #     module.up_proj = nn.Linear(
        #         module.hidden_size, STUDENT_INTERMEDIATE_SIZE)
        #     nn.init.normal_(module.up_proj.weight, mean=0.0, std=0.02)
        #     nn.init.zeros_(module.up_proj.bias)

        # if hasattr(module, 'gate_proj'):
        #     module.gate_proj = nn.Linear(
        #         module.hidden_size, STUDENT_INTERMEDIATE_SIZE)
        #     nn.init.normal_(module.gate_proj.weight, mean=0.0, std=0.02)
        #     nn.init.zeros_(module.gate_proj.bias)

        # if hasattr(module, 'down_proj'):
        #     module.down_proj = nn.Linear(
        #         STUDENT_INTERMEDIATE_SIZE, module.hidden_size)
        #     nn.init.normal_(module.down_proj.weight, mean=0.0, std=0.02)
        #     nn.init.zeros_(module.down_proj.bias)

# Freeze all parameters except MLP weights
for name, param in student_model.named_parameters():
    if "mlp" not in name:
        param.requires_grad = False

# Dataset class for function calling


class FunctionCallingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, tokenizer, max_length, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dataset with error handling
        print(f"Loading dataset: {dataset_name}")
        try:
            # Load the Glaive dataset
            dataset = load_dataset(dataset_name, streaming=True)
            split_name = "train" if "train" in dataset else list(dataset.keys())[0]

            # Convert to list with error handling
            self.data = []
            for i, item in enumerate(dataset[split_name]):
                print(i, item)
                if max_samples and i >= max_samples:
                    break
                try:
                    # Basic validation of item structure
                    if not isinstance(item, dict):
                        item = ast.literal_eval(item)

                    if "messages" not in item:
                        continue

                    # Convert Glaive format to our format
                    processed_item = {
                        "question": item["messages"],
                        "function": item.get("functions", []),
                    }
                    self.data.append(processed_item)
                except Exception as e:
                    print(f"Warning: Skipping item {i} due to error: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Falling back to dummy data for testing...")
            # Create dummy data for testing
            self.data = [
                {
                    "question": "What is the weather in San Francisco?",
                    "function": [
                        {
                            "name": "get_weather",
                            "description": "Get the current weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state",
                                    }
                                },
                                "required": ["location"],
                            },
                        }
                    ],
                }
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract messages and functions with better error handling
        messages = []
        question_data = item.get("question", None)

        if question_data is not None:
            try:
                if (
                    isinstance(question_data, dict)
                    and "role" in question_data
                    and "content" in question_data
                ):
                    messages = [question_data]
                elif isinstance(question_data, list):
                    if len(question_data) > 0:
                        if (
                            isinstance(question_data[0], dict)
                            and "role" in question_data[0]
                        ):
                            messages = question_data
                        elif (
                            isinstance(question_data[0], list)
                            and len(question_data[0]) > 0
                        ):
                            messages = question_data[0]
                elif isinstance(question_data, str):
                    messages = [{"role": "user", "content": question_data}]
            except Exception as e:
                print(f"Warning: Error processing question data: {str(e)}")
                messages = [{"role": "user", "content": str(question_data)}]

        # Extract functions with better error handling
        functions = []
        function_data = item.get("function", None)
        if function_data is not None:
            try:
                if isinstance(function_data, list):
                    functions = function_data
                elif isinstance(function_data, dict):
                    functions = [function_data]
                elif isinstance(function_data, str):
                    try:
                        parsed = json.loads(function_data)
                        functions = [parsed] if isinstance(parsed, dict) else parsed
                    except json.JSONDecodeError:
                        functions = [{"name": "function", "description": function_data}]
            except Exception as e:
                print(f"Warning: Error processing function data: {str(e)}")
                functions = [{"name": "function", "description": str(function_data)}]

        # Create system message with functions
        try:
            function_str = json.dumps(functions, indent=2) if functions else "{}"
            system_content = f"You are a helpful AI assistant that can use functions. Here are the available functions:\n\n{function_str}\n\nTo use a function, respond with the function name and parameters in this format: function_name(param1=value1, param2=value2)"
        except Exception as e:
            print(f"Warning: Error creating system message: {str(e)}")
            system_content = "You are a helpful AI assistant that can use functions."

        # Prepare chat messages
        chat = [{"role": "system", "content": system_content}]
        for msg in messages:
            try:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    chat.append(msg)
                elif isinstance(msg, str):
                    chat.append({"role": "user", "content": msg})
            except Exception as e:
                print(f"Warning: Error processing message: {str(e)}")
                continue

        # Convert to model input format
        try:
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template application failed: {str(e)}")
            prompt_parts = []
            for msg in chat:
                try:
                    role = msg.get("role", "").capitalize()
                    content = msg.get("content", "")
                    prompt_parts.append(f"{role}: {content}")
                except Exception as e:
                    print(f"Warning: Error formatting message: {str(e)}")
                    continue
            prompt = "\n".join(prompt_parts)

        # Tokenize with error handling
        try:
            encodings = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"Warning: Tokenization failed: {str(e)}")
            # Create dummy encodings
            encodings = {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.zeros(self.max_length, dtype=torch.long),
            }

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze().clone(),
        }


# Initialize dataset and dataloader
print("Initializing dataset...")
train_dataset = FunctionCallingDataset(
    DATASET_NAME, tokenizer, MAX_SEQ_LENGTH, max_samples=MAX_SAMPLES
)
print(train_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize optimizer and scheduler
optimizer = AdamW(
    filter(lambda p: p.requires_grad, student_model.parameters()), lr=LEARNING_RATE
)

total_training_steps = (
    len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS
) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=max(1, total_training_steps // 10),
    num_training_steps=total_training_steps,
)

# Training loop
print(f"Starting training for {NUM_EPOCHS} epochs...")
student_model.train()
progress_bar = tqdm(range(total_training_steps))

for epoch in range(NUM_EPOCHS):
    for step, batch in enumerate(train_dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden_states = teacher_outputs.hidden_states

        # Get student outputs
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_logits = student_outputs.logits
        student_hidden_states = student_outputs.hidden_states

        # Calculate losses
        # 1. Cross-entropy loss with ground truth
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 2. KL divergence loss with teacher logits
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / TEMPERATURE, dim=-1),
            F.softmax(teacher_logits / TEMPERATURE, dim=-1),
            reduction="batchmean",
        ) * (TEMPERATURE**2)

        # Combine losses
        loss = ALPHA_CE * ce_loss + ALPHA_KD * kd_loss

        # Scale loss for gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # Log progress
            if step % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, Step {step}, Loss: {loss.item():.4f}"
                )

    # Save checkpoint after each epoch
    checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch + 1}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    student_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved checkpoint to {checkpoint_dir}")

# Save final model
print("Saving final model...")
student_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete! Model saved to {OUTPUT_DIR}")
