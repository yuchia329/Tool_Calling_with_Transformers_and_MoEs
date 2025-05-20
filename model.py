from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, PretrainedConfig, PreTrainedModel
import torch.nn as nn
import torch

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
                         is_encoder_decoder=False, # Explicitly state it's not an encoder-decoder
                         is_decoder=True, # Explicitly state it's a decoder
                         **kwargs)

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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # if past is defined in model_inputs, then model is used in text generation
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        
        # The model_inputs dict defines the model inputs. Not all models accept all keyword arguments.
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "past_key_values": past_key_values, # Add this if your model's forward pass handles past_key_values
            # "use_cache": kwargs.get("use_cache", True) # Add this if your model's forward pass handles use_cache
        }
        # Note: If your StudentTransformerBlock and StudentModel.forward handle past_key_values (KV cache),
        # you should uncomment and properly manage "past_key_values" and "use_cache".
        # For now, this is a basic implementation that doesn't assume KV caching in your custom blocks.
        return model_inputs

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