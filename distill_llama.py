import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm
import json
from datetime import datetime

# Configuration
TEACHER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
STUDENT_INTERMEDIATE_SIZE = 2048  # Reduced from 8192
DATASET_NAME = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
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
device = torch.device("cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# Load teacher model
print("Loading teacher model...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)
teacher_model.eval()

# Create student model with modified MLP size
print("Creating student model...")
student_config = teacher_model.config
student_config.intermediate_size = STUDENT_INTERMEDIATE_SIZE
student_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    config=student_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)

# Randomly initialize MLP weights while keeping other weights from teacher
print("Randomly initializing MLP weights...")
for name, param in student_model.named_parameters():
    if "mlp" in name:
        # Initialize MLP weights with normal distribution
        if "weight" in name:
            nn.init.normal_(param, mean=0.0, std=0.02)
        elif "bias" in name:
            nn.init.zeros_(param)
    else:
        # Keep other weights from teacher model
        param.requires_grad = False

# Dataset class for function calling


class FunctionCallingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, tokenizer, max_length, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, streaming=True)
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        self.data = list(dataset[split_name].take(
            max_samples if max_samples else float('inf')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract messages and functions
        messages = []
        question_data = item.get("question", None)

        if question_data is not None:
            if isinstance(question_data, dict) and "role" in question_data and "content" in question_data:
                messages = [question_data]
            elif isinstance(question_data, list):
                if len(question_data) > 0:
                    if isinstance(question_data[0], dict) and "role" in question_data[0]:
                        messages = question_data
                    elif isinstance(question_data[0], list) and len(question_data[0]) > 0:
                        messages = question_data[0]
            elif isinstance(question_data, str):
                messages = [{"role": "user", "content": question_data}]

        # Extract functions
        functions = []
        function_data = item.get("function", None)
        if function_data is not None:
            if isinstance(function_data, list):
                functions = function_data
            elif isinstance(function_data, dict):
                functions = [function_data]
            elif isinstance(function_data, str):
                try:
                    parsed = json.loads(function_data)
                    functions = [parsed] if isinstance(
                        parsed, dict) else parsed
                except json.JSONDecodeError:
                    functions = [{"name": "function",
                                  "description": function_data}]

        # Create system message with functions
        function_str = json.dumps(functions, indent=2) if functions else "{}"
        system_content = f"You are a helpful AI assistant that can use functions. Here are the available functions:\n\n{function_str}\n\nTo use a function, respond with the function name and parameters in this format: function_name(param1=value1, param2=value2)"

        # Prepare chat messages
        chat = [{"role": "system", "content": system_content}]
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                chat.append(msg)
            elif isinstance(msg, str):
                chat.append({"role": "user", "content": msg})

        # Convert to model input format
        try:
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template application failed: {e}")
            prompt_parts = []
            for msg in chat:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt = "\n".join(prompt_parts)

        # Tokenize
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze().clone()
        }


# Initialize dataset and dataloader
print("Initializing dataset...")
train_dataset = FunctionCallingDataset(
    DATASET_NAME,
    tokenizer,
    MAX_SEQ_LENGTH,
    max_samples=MAX_SAMPLES
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Initialize optimizer and scheduler
optimizer = AdamW(
    filter(lambda p: p.requires_grad, student_model.parameters()),
    lr=LEARNING_RATE
)

total_training_steps = (len(train_dataloader) //
                        GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=max(1, total_training_steps // 10),
    num_training_steps=total_training_steps
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
                output_hidden_states=True
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden_states = teacher_outputs.hidden_states

        # Get student outputs
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        student_logits = student_outputs.logits
        student_hidden_states = student_outputs.hidden_states

        # Calculate losses
        # 1. Cross-entropy loss with ground truth
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # 2. KL divergence loss with teacher logits
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / TEMPERATURE, dim=-1),
            F.softmax(teacher_logits / TEMPERATURE, dim=-1),
            reduction="batchmean"
        ) * (TEMPERATURE ** 2)

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
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, Step {step}, Loss: {loss.item():.4f}")

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
