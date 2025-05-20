import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset # Assuming you have this
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, PretrainedConfig, PreTrainedModel
from tqdm.auto import tqdm
from datasets import load_dataset

# --- Configuration ---
TEACHER_MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat" # Changed to a Qwen chat model, often better for tool use
OUTPUT_DIR = "./distilled_student_ffn_toolace"
NUM_EPOCHS = 3
BATCH_SIZE = 2 # Reduced batch size, data can be longer
LEARNING_RATE = 5e-5
DISTILLATION_TEMP = 2.0
ALPHA_CE = 0.5
ALPHA_FFN_KD = 0.5
MAX_SEQ_LENGTH = 512 # Define a max sequence length for data
TOOL_CALLING_DATASET_NAME = "Team-ACE/ToolACE" # Changed to ToolACE dataset
MAX_DATASET_SAMPLES = 1000 # For faster experimentation, limit samples

# --- 1. Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load Tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token_id} ({tokenizer.pad_token})")
# Ensure pad_token_id is correctly set in config for StudentModel if tokenizer.pad_token was None initially
student_pad_token_id = tokenizer.pad_token_id


# --- 3. Load Teacher Model (ensure it outputs hidden states) ---
print(f"Loading teacher model: {TEACHER_MODEL_NAME}...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_NAME,
    trust_remote_code=True,
    output_hidden_states=True
).to(device)
teacher_model.eval()

# --- 4. Define Student Model (with focus on FFNs) ---
class StudentConfig(PretrainedConfig):
    model_type = "student_ffn_distill"
    def __init__(self, vocab_size=50257, n_positions=512, n_embd=384, n_layer=6, n_head=6,
                 n_inner=None, activation_function="gelu_new", resid_pdrop=0.1,
                 embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-5,
                 initializer_range=0.02, output_hidden_states=True, # Student must output hidden states
                 teacher_hidden_size=None, # To store teacher's hidden dim for projection
                 # Add explicit token ID parameters
                 pad_token_id=None,
                 bos_token_id=None,
                 eos_token_id=None,
                 **kwargs):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.output_hidden_states = output_hidden_states # Ensure student outputs hidden states
        self.teacher_hidden_size = teacher_hidden_size # For FFN projection
        # Use the passed-in token IDs for the super() call
        super().__init__(pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         **kwargs)

class StudentTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # MultiHeadAttention - this is what we want to "keep the same" or train lightly
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.attn_pdrop, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # FFN/MLP - this is what we primarily want to distill
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            nn.GELU(), # Consider using config.activation_function
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        # Optional: Projection layer if student n_embd != teacher_hidden_size for FFN output
        self.ffn_output_projection = None
        if config.teacher_hidden_size and config.n_embd != config.teacher_hidden_size:
            self.ffn_output_projection = nn.Linear(config.n_embd, config.teacher_hidden_size, bias=False)

    def forward(self, x, attention_mask_mha=None, is_causal=False):
        # For causal LM, MHA needs a causal mask or is_causal=True if supported by PyTorch version & MHA
        # PyTorch MHA's is_causal is for self-attention.
        # key_padding_mask (True for pad) is also important.
        
        # Build causal mask if needed and not handled by MHA's is_causal
        causal_mask = None
        if is_causal:
            seq_len = x.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        attn_output, _ = self.attn(x, x, x, attn_mask=causal_mask, key_padding_mask=attention_mask_mha, need_weights=False)
        x = x + attn_output
        x = self.ln_1(x)
        ffn_hidden_state = self.mlp(x)
        x = x + ffn_hidden_state # Residual 2
        block_output = self.ln_2(x) # Output of the full block

        # For distillation, we might want the output of the FFN *before* residual/LN, or the block_output
        # Let's use block_output for distillation, assuming it captures FFN's transformation.
        # If projection is needed for distillation against teacher:
        projected_ffn_output_for_distill = block_output
        if self.ffn_output_projection:
            projected_ffn_output_for_distill = self.ffn_output_projection(block_output)
        return block_output, projected_ffn_output_for_distill

class StudentModel(PreTrainedModel):
    config_class = StudentConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([StudentTransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def _init_weights(self, module): # Renamed from init_weights for clarity
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, labels=None, output_attentions=False, output_hidden_states=None):
        # output_hidden_states argument in forward overrides config.output_hidden_states
        # Ensure it's True if we need hidden states for distillation, regardless of call-time override
        _output_hidden_states = True # Force true for internal logic

        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        all_layer_outputs_for_distill = [] # To store (projected) outputs of each block for FFN distillation
        all_raw_hidden_states = [] # To store raw outputs of each student layer

        # Prepare attention mask for MHA: True for padding, False for non-padding
        mha_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        # Causal mask is typically handled inside MHA or by preparing a combined mask.
        # For simplicity, we assume StudentTransformerBlock's MHA handles causality or gets an appropriate mask.

        if _output_hidden_states:
            all_raw_hidden_states.append(hidden_states) # Input embeddings state

        for block in self.h:
            hidden_states, projected_output_for_distill = block(hidden_states, attention_mask_mha=mha_key_padding_mask, is_causal=True) # Pass is_causal=True
            if _output_hidden_states:
                all_raw_hidden_states.append(hidden_states) # Raw output of this student block
                all_layer_outputs_for_distill.append(projected_output_for_distill)
        final_hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(final_hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states_for_distill": tuple(all_layer_outputs_for_distill) if _output_hidden_states else None,
            "hidden_states": tuple(all_raw_hidden_states) if _output_hidden_states else None, # All raw student hidden states
        }

# --- Initialize Student Model ---
print("Initializing student model...")
student_config = StudentConfig(
    vocab_size=teacher_model.config.vocab_size,
    pad_token_id=student_pad_token_id, # Use the potentially updated pad_token_id
    bos_token_id=getattr(tokenizer, 'bos_token_id', None),
    eos_token_id=getattr(tokenizer, 'eos_token_id', None),
    n_positions=MAX_SEQ_LENGTH, # Ensure student can handle the sequence length
    n_embd=teacher_model.config.hidden_size // 2 if hasattr(teacher_model.config, 'hidden_size') else 384,
    n_layer=teacher_model.config.num_hidden_layers // 2 if hasattr(teacher_model.config, 'num_hidden_layers') else 6,
    n_head=teacher_model.config.num_attention_heads // 2 if hasattr(teacher_model.config, 'num_attention_heads') else 6,
    n_inner=(teacher_model.config.intermediate_size // 2) if hasattr(teacher_model.config, 'intermediate_size') else (4*(teacher_model.config.hidden_size // 2)),
    output_hidden_states=True,
    teacher_hidden_size=teacher_model.config.hidden_size
)
student_model = StudentModel(student_config).to(device)
print(f"Student model initialized with {student_config.n_layer} layers, hidden_size {student_config.n_embd}.")

# --- 5. Optimizer ---
optimizer = AdamW(filter(lambda p: p.requires_grad, student_model.parameters()), lr=LEARNING_RATE)
print(f"Optimizer with single LR={LEARNING_RATE} for all trainable parameters.")

# --- Load and Preprocess Tool Calling Dataset ---
def load_and_preprocess_tool_calling_dataset(dataset_name, tokenizer, max_seq_length, split="train", max_samples=None):
    print(f"Loading dataset: {dataset_name}, split: {split}")
    try:
        # For ToolACE, it might be necessary to specify a subset/configuration if it has multiple.
        # Trying to load it directly. If it fails, one might need to check its specific loading instructions on Hugging Face Hub.
        dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}': {e}")
        print("Please ensure the 'datasets' library is installed (pip install datasets), you have internet access, and the dataset name/split are correct.")
        print("If the dataset has specific configurations (e.g., ToolACE/generation), you might need to specify it in dataset_name or as a second argument to load_dataset.")
        print("Falling back to dummy data if training is attempted.")
        return None

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    print("Preprocessing dataset...")
    for example in tqdm(dataset, desc=f"Processing {dataset_name} Data"):
        user_turn_content = None
        target_tool_call = None

        # Try common structures for tool calling datasets:
        # 1. Conversational format (like Gorilla)
        conversations = example.get('conversations')
        if conversations and isinstance(conversations, list) and len(conversations) >= 2:
            if conversations[0].get('from', '').lower() == 'user' and \
               conversations[1].get('from', '').lower() in ['gpt', 'assistant']:
                user_turn_content = conversations[0].get('value')
                target_tool_call = conversations[1].get('value')
            elif len(conversations) > 1 and conversations[0].get('from', '').lower() == 'system' and \
                 conversations[1].get('from', '').lower() == 'user' and len(conversations) > 2 and \
                 conversations[2].get('from', '').lower() in ['gpt', 'assistant']:
                user_turn_content = conversations[1].get('value')
                target_tool_call = conversations[2].get('value')
        
        # 2. Instruction/Output or similar direct fields (common in ToolAlpaca, etc.)
        if user_turn_content is None or target_tool_call is None:
            # Check for ToolACE specific fields or common alternatives
            # Based on a quick look at ToolACE, it might have 'query' and 'response_with_tool' or similar
            # Or it might be in a more complex JSON structure within a field.
            # For now, let's try general instruction/output fields, then specific ToolACE like fields if known.
            # This part will likely need adjustment based on the *exact* structure of ToolACE samples.
            if 'instruction' in example and 'output' in example:
                user_turn_content = example['instruction']
                target_tool_call = example['output']
            elif 'query' in example and 'response_with_tool' in example: # Hypothetical for ToolACE
                user_turn_content = example['query']
                target_tool_call = example['response_with_tool']
            elif 'query' in example and 'tool_code' in example: # Another hypothetical
                user_turn_content = example['query']
                target_tool_call = example['tool_code']
            elif 'prompt' in example and 'completion' in example:
                 user_turn_content = example['prompt']
                 target_tool_call = example['completion']
             # Add more elif branches here if you discover the specific field names for ToolACE

        if not user_turn_content or not isinstance(user_turn_content, str) or \
           not target_tool_call or not isinstance(target_tool_call, str):
            print(f"Skipping example due to missing/invalid content for user query or tool call. Example ID (if available): {example.get('id', 'N/A')}")
            # print(f"Problematic example content: User: {user_turn_content}, Target: {target_tool_call}")
            continue

        tokenized_input = tokenizer(
            user_turn_content,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        tokenized_labels = tokenizer(
            target_tool_call,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        tokenized_labels[tokenized_labels == tokenizer.pad_token_id] = -100

        input_ids_list.append(tokenized_input.input_ids.squeeze(0))
        attention_mask_list.append(tokenized_input.attention_mask.squeeze(0))
        labels_list.append(tokenized_labels.squeeze(0))

    if not input_ids_list:
        print("No data processed. Check dataset format or processing logic for the selected dataset.")
        return None

    return TensorDataset(
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
        torch.stack(labels_list)
    )

# --- Prepare DataLoader ---
# Replace dummy data with the chosen tool calling dataset
train_dataset = load_and_preprocess_tool_calling_dataset(
    TOOL_CALLING_DATASET_NAME,
    tokenizer,
    MAX_SEQ_LENGTH,
    split="train",
    max_samples=MAX_DATASET_SAMPLES
)

if train_dataset:
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset '{TOOL_CALLING_DATASET_NAME}' loaded with {len(train_dataset)} samples. Batch size: {BATCH_SIZE}")
else:
    print(f"Failed to load dataset '{TOOL_CALLING_DATASET_NAME}'. Exiting or add fallback to dummy data if you want to continue.")
    # Fallback to dummy data for testing if Gorilla loading fails (optional)
    print("Preparing dummy dataset as fallback...")
    num_samples = 128
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (num_samples, MAX_SEQ_LENGTH), device=device, dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids, device=device)
    for i in range(num_samples // 4):
        pad_len = torch.randint(MAX_SEQ_LENGTH // 4, MAX_SEQ_LENGTH // 2, (1,)).item()
        dummy_input_ids[i, -pad_len:] = tokenizer.pad_token_id
        dummy_attention_mask[i, -pad_len:] = 0
    dummy_labels = dummy_input_ids.clone()
    dummy_labels[dummy_labels == tokenizer.pad_token_id] = -100 # For CE loss
    dataset = TensorDataset(dummy_input_ids, dummy_attention_mask, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dummy dataset ready with {len(dataset)} samples. Batch size: {BATCH_SIZE}")


num_training_steps = NUM_EPOCHS * len(dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=max(1, num_training_steps // 10), num_training_steps=num_training_steps
)

# --- 6. FFN Distillation Loss Function ---
def compute_ffn_distillation_loss(student_block_outputs, teacher_hidden_states, attention_mask):
    """
    Computes MSE loss between student's (projected) block outputs and teacher's hidden states.
    student_block_outputs: Tuple of student's layer outputs, already projected to teacher's dimension.
    teacher_hidden_states: Tuple from teacher_model (embeddings, layer1_out, ...).
    attention_mask: Original (batch_size, seq_length) mask, 1 for non-pad, 0 for pad.
    """
    ffn_loss = 0.0
    if not student_block_outputs or not teacher_hidden_states:
        return torch.tensor(0.0).to(device)
    num_student_layers_distill = len(student_block_outputs)
    # Teacher hidden_states include embedding output, so N layers means N+1 states.
    # We match student layer i with teacher layer i (output of i-th transformer block).
    num_teacher_layers_available = len(teacher_hidden_states) - 1
    distill_count = 0
    for i in range(num_student_layers_distill):
        teacher_idx = i + 1
        if teacher_idx >= len(teacher_hidden_states):
            continue
        s_hidden = student_block_outputs[i]
        t_hidden = teacher_hidden_states[teacher_idx].detach()
        if s_hidden.shape[-1] != t_hidden.shape[-1]: # Check if projection was supposed to happen but didn't match
             print(f"Warning: Shape mismatch in FFN distillation after potential projection! Layer {i}: Student {s_hidden.shape}, Teacher {t_hidden.shape}. Ensure student's ffn_output_projection is correct or teacher_hidden_size is set properly in StudentConfig.")
             # Attempt to project teacher if student is smaller and no projection layer exists on student.
             # This is a fallback, ideally student projects to teacher's dim.
             if s_hidden.shape[-1] < t_hidden.shape[-1] and not student_model.h[i].ffn_output_projection:
                 temp_projection = nn.Linear(t_hidden.shape[-1], s_hidden.shape[-1], bias=False).to(device)
                 t_hidden = temp_projection(t_hidden)
             elif s_hidden.shape[-1] > t_hidden.shape[-1] and not student_model.h[i].ffn_output_projection: # Student larger, teacher needs projection
                 temp_projection = nn.Linear(s_hidden.shape[-1], t_hidden.shape[-1], bias=False).to(device) # Project student instead
                 s_hidden = temp_projection(s_hidden)
             else: # Shapes mismatch and either projection exists or no clear way to project
                 continue


        if s_hidden.shape != t_hidden.shape:
             print(f"Shape mismatch in FFN distillation! Layer {i}: Student {s_hidden.shape}, Teacher {t_hidden.shape}. Skipping.")
             continue

        mask = attention_mask.unsqueeze(-1).expand_as(s_hidden).float()
        # Count non-padded elements for averaging
        elements_to_consider = mask.sum()
        if elements_to_consider > 0:
            loss_layer = (F.mse_loss(s_hidden * mask, t_hidden * mask, reduction='sum')) / elements_to_consider
            ffn_loss += loss_layer
            distill_count +=1
    return ffn_loss / distill_count if distill_count > 0 else torch.tensor(0.0).to(device)

# --- 7. Training Loop ---
print("Starting FFN-focused distillation training...")
if not hasattr(dataloader, '__len__') or len(dataloader) == 0:
    print("Dataloader is empty. Cannot start training.")
else:
    student_model.train()
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
        total_epoch_loss = 0
        total_epoch_ce_loss = 0
        total_epoch_ffn_kd_loss = 0

        for batch_idx, batch in enumerate(progress_bar):
            input_ids, attention_mask, labels_for_ce = [b.to(device) for b in batch]

            # Teacher forward pass (only on input_ids, not labels_for_ce)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_all_hidden_states = teacher_outputs.hidden_states

            # Student forward pass
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_for_ce # Pass the tokenized tool call as labels for CE loss
            )
            student_final_ce_loss = student_outputs["loss"]
            student_hidden_states_for_distill = student_outputs["hidden_states_for_distill"]

            loss_ffn_kd = torch.tensor(0.0).to(device)
            if ALPHA_FFN_KD > 0 and student_hidden_states_for_distill and teacher_all_hidden_states:
                loss_ffn_kd = compute_ffn_distillation_loss(
                    student_hidden_states_for_distill,
                    teacher_all_hidden_states,
                    attention_mask
                )

            current_loss = torch.tensor(0.0, device=device) # Ensure current_loss is a tensor
            valid_loss_added = False
            if student_final_ce_loss is not None and ALPHA_CE > 0:
                current_loss += ALPHA_CE * student_final_ce_loss
                valid_loss_added = True
            if ALPHA_FFN_KD > 0:
                current_loss += ALPHA_FFN_KD * loss_ffn_kd
                valid_loss_added = True
            
            if not valid_loss_added: # If no loss components were added (e.g. all alphas are 0)
                # print("Warning: No loss components active. Gradients will not be computed.")
                # To avoid error with backward() on a non-leaf tensor with requires_grad=False if current_loss is still 0.0
                # One option is to skip optim step or ensure loss is always a tensor that requires grad if it's > 0
                 pass # Allow it to proceed, backward() on a zero tensor might be fine or might need dummy loss.


            optimizer.zero_grad()
            if isinstance(current_loss, torch.Tensor) and current_loss.requires_grad:
                 current_loss.backward()
                 torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                 optimizer.step()
            elif current_loss.item() != 0 : # If it's a non-zero tensor but doesn't require grad (should not happen if built from losses)
                 print(f"Warning: Loss tensor does not require grad: {current_loss.item()}")

            lr_scheduler.step()

            total_epoch_loss += current_loss.item() if isinstance(current_loss, torch.Tensor) else 0.0
            total_epoch_ce_loss += student_final_ce_loss.item() if student_final_ce_loss is not None else 0.0
            total_epoch_ffn_kd_loss += loss_ffn_kd.item() if isinstance(loss_ffn_kd, torch.Tensor) else 0.0

            progress_bar.set_postfix({
                "Loss": f"{current_loss.item():.4f}" if isinstance(current_loss, torch.Tensor) else "N/A",
                "CE": f"{student_final_ce_loss.item():.4f}" if student_final_ce_loss is not None else "N/A",
                "FFN_KD": f"{loss_ffn_kd.item():.4f}" if isinstance(loss_ffn_kd, torch.Tensor) else "N/A",
                "LR": f"{lr_scheduler.get_last_lr()[0]:.3e}"
            })
            
        avg_epoch_loss = total_epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_epoch_ce_loss = total_epoch_ce_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_epoch_ffn_kd_loss = total_epoch_ffn_kd_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_epoch_loss:.4f}, Avg CE: {avg_epoch_ce_loss:.4f}, Avg FFN_KD: {avg_epoch_ffn_kd_loss:.4f}")

# --- 8. Save Student Model ---
print(f"\nTraining complete. Saving student model to {OUTPUT_DIR}...")
student_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Student model and tokenizer saved.")