import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset, load_from_disk

from tqdm.auto import tqdm
import json
import os

# --- Configuration ---
TEACHER_MODEL_NAME = "meta-llama/Llama-3.2-3B"
STUDENT_MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
# TRYING A SPECIFIC CONFIG - VERIFY THIS ON HF HUB for Berkeley-Function-Calling-Leaderboard
DATASET_CONFIG = None
OUTPUT_DIR = "./distilled_llama3.2_1b_on_berkeley"

NUM_EPOCHS = 3
BATCH_SIZE = 1  # Start with 1 due to model sizes; increase if VRAM allows with gradient accumulation
# Effective batch size will be BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 1024  # Adjust based on dataset and VRAM
# Max samples from dataset to use for training/eval. None for all.
MAX_SAMPLES = 2000
EVAL_SAMPLES = 200   # Max samples for evaluation periodically

# Distillation HPs
TEMPERATURE = 2.0      # Temperature for softening teacher logits
# Weight for Cross-Entropy loss (hard labels from dataset)
ALPHA_CE = 0.4
# Weight for KL Divergence distillation loss (soft labels from teacher)
ALPHA_KD = 0.6
# ALPHA_HIDDEN = 0.1   # Optional: Weight for hidden state matching loss (not implemented in this version)

# Hugging Face Token (if required for Llama models)
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Or set your token here directly

# --- 1. Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA Memory Available: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")

# --- 2. Load Tokenizer ---
# Llama 3.2 models should ideally use the same tokenizer or highly compatible ones.
# Using the student's tokenizer as it's the one we are fine-tuning.
print(f"Loading tokenizer for {STUDENT_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(
    STUDENT_MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(
        f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# --- 3. Load Models ---
print(f"Loading teacher model: {TEACHER_MODEL_NAME}...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    trust_remote_code=True
).to(device)
teacher_model.eval()
print("Teacher model loaded.")

print(f"Loading student model: {STUDENT_MODEL_NAME}...")
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    trust_remote_code=True
).to(device)
print("Student model loaded.")

if student_model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
    student_model.config.pad_token_id = tokenizer.pad_token_id

# --- 4. Load & Preprocess Dataset ---


class ToolCallingDataset(Dataset):
    def __init__(self, dataset_name, dataset_config, tokenizer, max_seq_length, split="train", max_samples=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.input_texts = []
        self.target_texts = []
        self.sample_ids = []

        print(
            f"Loading and preprocessing dataset: {dataset_name} (config: {dataset_config}), split: {split}")
        raw_dataset_iterable = None
        try:
            raw_dataset_iterable = load_dataset(
                "json", data_files={"train": "Berkeley-Function-Calling-Leaderboard/BFCL_v3_exec_multiple.json"}, split="train")

        except Exception as e:
            raise e
        processed_count = 0
        print("Iterating through dataset samples...")
        for i, example in enumerate(tqdm(raw_dataset_iterable, desc="Processing Dataset Samples")):
            # print(i, example) # DEBUG: Keep for debugging if needed, but remove for production
            if max_samples is not None and processed_count >= max_samples:
                break

            sample_id = example.get("id", f"sample_{i}")

            # 1. Extract tools (functions)
            # Expected: list of tool definition dicts
            tools_list_of_dicts = example.get("function", None)

            # 2. Extract chat messages
            actual_messages = None
            question_field = example.get("question")

            if isinstance(question_field, list) and len(question_field) > 0:
                # Prioritize the first conversation in the list: question_field[0]
                first_conversation_candidate = question_field[0]
                if isinstance(first_conversation_candidate, list) and \
                   all(isinstance(turn, dict) and "role" in turn and "content" in turn for turn in first_conversation_candidate):
                    actual_messages = first_conversation_candidate
                # Fallback: if question_field itself is a flat list of messages (less common based on samples)
                elif all(isinstance(turn, dict) and "role" in turn and "content" in turn for turn in question_field):
                    actual_messages = question_field

            if not actual_messages:  # If not extracted from "question"
                chat_history_fallback = example.get("chat_history")
                if isinstance(chat_history_fallback, list) and \
                   all(isinstance(turn, dict) and "role" in turn and "content" in turn for turn in chat_history_fallback):
                    actual_messages = chat_history_fallback
                else:
                    user_query_text_fallback = example.get(
                        "query", example.get("prompt", example.get("instruction")))
                    if user_query_text_fallback:
                        actual_messages = [
                            {"role": "user", "content": str(user_query_text_fallback)}]
                        # If using this fallback, tools might need to be manually added to system prompt if not handled by template
                    else:
                        print(
                            f"  Skipping sample {sample_id}: Could not extract valid messages from 'question', 'chat_history', or other text fields.")
                        continue

            if not actual_messages:  # Safeguard, should be caught above
                print(
                    f"  Skipping sample {sample_id}: No processable message content found after all fallbacks.")
                continue

            # 3. Extract target answer string (optional)
            answer_str = None
            ground_truth_field = example.get("ground_truth")
            if isinstance(ground_truth_field, list) and len(ground_truth_field) > 0 and ground_truth_field[0]:
                answer_str = str(ground_truth_field[0])
            elif isinstance(ground_truth_field, str) and ground_truth_field:
                answer_str = ground_truth_field

            if not answer_str:  # Fallback to "answer" field
                answer_field = example.get("answer")
                if isinstance(answer_field, str) and answer_field:
                    answer_str = answer_field

            if not answer_str:
                print(
                    f"  Skipping sample {sample_id}: No 'ground_truth' or 'answer' found for target.")
                continue

            # 4. Construct input prompt using tokenizer's chat template
            input_prompt = None
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    # Pass tools directly if the tokenizer/template supports it (e.g., Llama 3)
                    input_prompt = self.tokenizer.apply_chat_template(
                        actual_messages,
                        tools=tools_list_of_dicts,  # Will be None if not found
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except TypeError as te:  # Catch if 'tools' kwarg is unexpected or other issues
                    # print(f"  Info for sample {sample_id}: tokenizer.apply_chat_template raised TypeError ('{te}'). Retrying without 'tools' kwarg.")
                    # Fallback: Try without the 'tools' parameter.
                    # Manually create a system prompt with tools if not already present in actual_messages.
                    messages_for_fallback_template = [
                        m for m in actual_messages]  # Make a copy
                    has_system_message_with_tools = False
                    if any(m['role'] == 'system' for m in messages_for_fallback_template):
                        # rudimentary check, could be improved to see if tools are actually in the system msg
                        has_system_message_with_tools = True

                    if tools_list_of_dicts and not has_system_message_with_tools:
                        # Create a generic system message with tools if none exists or if existing one doesn't seem to have them.
                        # This is a simplified approach.
                        tool_desc_str = "You are a helpful assistant. Here are the available tools:\n" + \
                            json.dumps(tools_list_of_dicts, indent=2)
                        # Check if there is any system message
                        system_message_exists = any(
                            m['role'] == 'system' for m in messages_for_fallback_template)
                        if not system_message_exists:
                            messages_for_fallback_template.insert(
                                0, {"role": "system", "content": tool_desc_str})
                        # If system message exists but we are in this fallback, it implies tools weren't handled by `tools=`
                        # We might choose to append to existing system message or add a new one.
                        # For simplicity, if system message exists, assume it's okay or user prepared it.
                        # A more robust solution would involve checking/merging tool descriptions.

                    try:
                        input_prompt = self.tokenizer.apply_chat_template(
                            messages_for_fallback_template,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    except Exception as e_fallback:
                        print(
                            f"  Skipping sample {sample_id}: Failed to apply chat template in fallback: {e_fallback}")
                        continue
                except Exception as e:  # Catch other unexpected errors with apply_chat_template
                    print(
                        f"  Skipping sample {sample_id}: Error applying chat template: {e}")
                    continue
            else:
                # Manual prompt construction (legacy, if tokenizer doesn't have apply_chat_template)
                print(
                    f"  Warning for sample {sample_id}: tokenizer has no apply_chat_template. Using manual prompt construction.")
                input_prompt_parts = []
                if tools_list_of_dicts:
                    input_prompt_parts.append(
                        "<TOOLS_AVAILABLE>\n" + json.dumps(tools_list_of_dicts, indent=2) + "\n</TOOLS_AVAILABLE>")
                for msg in actual_messages:
                    input_prompt_parts.append(
                        f"{msg['role'].capitalize()}: {msg['content']}")
                # Prompt for generation
                input_prompt_parts.append("Assistant:")
                input_prompt = "\\n".join(input_prompt_parts)

            if not input_prompt:
                # This case should ideally be caught by 'continue' statements above
                print(
                    f"  Skipping sample {sample_id}: input_prompt could not be constructed.")
                continue

            self.input_texts.append(input_prompt)
            self.target_texts.append(answer_str)
            self.sample_ids.append(sample_id)
            processed_count += 1  # Increment after successful processing of a sample

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        tokenized_input = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tokenized_target = self.tokenizer(
            target_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tokenized_target.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized_input.input_ids.squeeze(0),
            "attention_mask": tokenized_input.attention_mask.squeeze(0),
            "labels": labels,
            "id": self.sample_ids[idx]
        }


print("\nInitializing training dataset...")
train_dataset = ToolCallingDataset(
    DATASET_NAME, DATASET_CONFIG, tokenizer, MAX_SEQ_LENGTH, split=None, max_samples=MAX_SAMPLES
)
# print(f"Training dataset length: {train_dataset}")

# if not train_dataset or len(train_dataset) == 0:
#     raise ValueError(
#         "Training dataset is empty. Check data loading and preprocessing.")

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)
total_training_steps = (len(train_dataloader) //
                        GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=max(1, total_training_steps // 10),
    num_training_steps=total_training_steps
)
print(
    f"Total training steps (considering accumulation): {total_training_steps}")


def compute_distillation_loss(student_logits, teacher_logits, hard_labels, temp, alpha_ce, alpha_kd, pad_token_id):
    active_hard_labels_mask = hard_labels != -100
    loss_ce = torch.tensor(0.0, device=student_logits.device)
    if active_hard_labels_mask.any():
        loss_ce = F.cross_entropy(
            student_logits[active_hard_labels_mask],
            hard_labels[active_hard_labels_mask],
            ignore_index=-100
        )

    soft_teacher_logits = (teacher_logits / temp).detach()
    soft_student_logits = student_logits / temp

    loss_kd = torch.tensor(0.0, device=student_logits.device)
    if active_hard_labels_mask.any():
        log_p_teacher = F.log_softmax(
            soft_teacher_logits[active_hard_labels_mask], dim=-1)
        p_student = F.softmax(
            soft_student_logits[active_hard_labels_mask], dim=-1)
        loss_kd = F.kl_div(log_p_teacher, p_student,
                           reduction='batchmean', log_target=True) * (temp ** 2)

    total_loss = alpha_ce * loss_ce + alpha_kd * loss_kd
    return total_loss, loss_ce, loss_kd


print("\n--- Starting Distillation Training ---")
student_model.train()
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

global_step = 0
for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    epoch_total_loss = 0
    epoch_ce_loss = 0
    epoch_kd_loss = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits_full = teacher_outputs.logits

            student_outputs = student_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=None)
            student_logits_full = student_outputs.logits

            shift_teacher_logits = teacher_logits_full[...,
                                                       :-1, :].contiguous()
            shift_student_logits = student_logits_full[...,
                                                       :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            teacher_logits_flat = shift_teacher_logits.view(
                -1, shift_teacher_logits.size(-1))
            student_logits_flat = shift_student_logits.view(
                -1, shift_student_logits.size(-1))
            labels_flat = shift_labels.view(-1)

            loss, loss_ce, loss_kd = compute_distillation_loss(
                student_logits=student_logits_flat,
                teacher_logits=teacher_logits_flat,
                hard_labels=labels_flat,
                temp=TEMPERATURE,
                alpha_ce=ALPHA_CE,
                alpha_kd=ALPHA_KD,
                pad_token_id=tokenizer.pad_token_id
            )
            current_loss_for_accumulation = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(current_loss_for_accumulation).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
            global_step += 1

            epoch_total_loss += loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_kd_loss += loss_kd.item()

            if global_step % (max(1, len(train_dataloader) // (GRADIENT_ACCUMULATION_STEPS * 10))) == 0:
                print(
                    f"  Step: {global_step}/{total_training_steps}, Batch: {batch_idx+1}/{len(train_dataloader)}, LR: {lr_scheduler.get_last_lr()[0]:.2e}, Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, KD: {loss_kd.item():.4f})")

    avg_epoch_loss = epoch_total_loss / \
        (len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    avg_epoch_ce_loss = epoch_ce_loss / \
        (len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    avg_epoch_kd_loss = epoch_kd_loss / \
        (len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_epoch_loss:.4f} (Avg CE: {avg_epoch_ce_loss:.4f}, Avg KD: {avg_epoch_kd_loss:.4f})")

    if (epoch + 1) % 1 == 0:
        epoch_output_dir = os.path.join(
            OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {epoch_output_dir}")
        student_model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)

print(f"\nTraining complete. Saving final student model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
student_model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
print("Final student model and tokenizer saved.")

print("\nDistillation script finished.")
