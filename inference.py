from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Assuming StudentConfig and StudentModel are defined in 'test.py' in the same directory
# If they are in a different file, adjust the import accordingly (e.g., from .student_module import StudentConfig, StudentModel)
from model import StudentConfig, StudentModel 

# Register your custom classes
AutoConfig.register("student_ffn_distill", StudentConfig)
AutoModelForCausalLM.register(StudentConfig, StudentModel)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

MODEL_PATH = "./distilled_student_ffn_toolace" # Define path for clarity and reuse

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, # Use the defined path
    torch_dtype="auto",
    trust_remote_code=True # Add if your custom student model code is needed
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, # Load tokenizer from the same path
    trust_remote_code=True # Add if custom tokenizer code might be involved (less common)
)

# Set pad_token to eos_token if not already set, common for Qwen and helps avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Ensure model inputs are on the same device as the model
model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask, # Explicitly pass attention_mask
    max_new_tokens=512,
    pad_token_id=tokenizer.pad_token_id # Pass pad_token_id to generate
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)