import os
import torch
import time
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer
from statistics import mean
from datetime import datetime

# Check for MPS (Apple Silicon GPU) availability
MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Set device for computation
device = get_device()
print(f"Using device: {device}")

# Memory-optimized evaluation settings for macOS
if device.type == "mps":
    # MPS-specific settings - adjust batch sizes and lengths for 48GB unified memory
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_SAMPLE_SIZE = 50
    print(f"Configured for MPS device with optimized memory settings")
    # Set environment variable to control MPS memory usage
    # Use minimal preallocated memory
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
else:
    # Regular settings for CUDA or CPU
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_SAMPLE_SIZE = 100

# Define a custom dataset class to format OpenOrca data for Llama-3.2-3B


class OpenOrcaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, include_labels=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get item from dataset
        item = self.dataset[idx]

        # Format the prompt according to Llama-3.2 chat format
        if item['system_prompt'] and len(item['system_prompt'].strip()) > 0:
            chat = [
                {"role": "system", "content": item['system_prompt']},
                {"role": "user", "content": item['question']},
            ]
            if self.include_labels:
                chat.append({"role": "assistant", "content": item['response']})
        else:
            chat = [
                {"role": "user", "content": item['question']},
            ]
            if self.include_labels:
                chat.append({"role": "assistant", "content": item['response']})

        # Convert chat to tokenizer format
        try:
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template application failed with error: {e}")
            print("Falling back to manual formatting...")
            # Manual fallback formatting
            prompt_parts = []
            for message in chat:
                if message['role'] == 'system':
                    prompt_parts.append(f"System: {message['content']}")
                elif message['role'] == 'user':
                    prompt_parts.append(f"User: {message['content']}")
                elif message['role'] == 'assistant':
                    prompt_parts.append(f"Assistant: {message['content']}")
            prompt = "\n".join(prompt_parts)
            # If we're supposed to generate, add the assistant prompt
            if self.include_labels is False:
                prompt += "\nAssistant:"

        # Tokenize the prompt
        encodings = self.tokenizer(
            prompt, max_length=self.max_length, padding="max_length", truncation=True)

        # Create input_ids and attention_mask
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'original_question': item['question'],
            'original_response': item['response'],
        }

        if self.include_labels:
            # Use same IDs for labels in perplexity evaluation
            result['labels'] = input_ids.clone()

        return result


def build_openorca_dataset(sample_size=None, max_length=None, batch_size=None, include_labels=True):
    """
    Build a dataset from Open-Orca/OpenOrca for inference with meta-llama/Llama-3.2-3B model.

    Args:
        sample_size: Number of examples to include in the dataset (uses DEFAULT_SAMPLE_SIZE if None)
        max_length: Maximum sequence length for tokenization (uses DEFAULT_MAX_LENGTH if None)  
        batch_size: Batch size for dataloader (uses DEFAULT_BATCH_SIZE if None)
        include_labels: Whether to include labels in the dataset (for training or perplexity)

    Returns:
        train_dataloader: DataLoader for training data (or None if train data not needed)
        eval_dataloader: DataLoader for evaluation data
        tokenizer: The tokenizer used
    """
    # Use device-specific defaults if values not provided
    if sample_size is None:
        sample_size = DEFAULT_SAMPLE_SIZE
    if max_length is None:
        max_length = DEFAULT_MAX_LENGTH
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE

    print(
        f"Building OpenOrca dataset with sample_size={sample_size}, max_length={max_length}, batch_size={batch_size}")
    print("Loading OpenOrca dataset...")
    # Load OpenOrca dataset
    dataset = load_dataset("Open-Orca/OpenOrca", streaming=True)

    # Convert streaming dataset to list to sample from it
    print("Sampling dataset...")
    dataset = list(dataset["train"].take(sample_size))

    # Split dataset into train and eval sets (90% train, 10% eval)
    # train_size = int(0.01 * len(dataset))
    # train_dataset = dataset[:train_size]
    # eval_dataset = dataset[train_size:]
    eval_dataset = dataset

    # print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Load Llama-3.2-3B tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        print("Pad token not found, setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(
            f"Set pad token to: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # Check if chat template is set, if not, set a default Llama-3 chat template
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("Chat template not found, setting default Llama-3 chat template...")
        # Default Llama-3 chat template (simplification of the actual template)
        DEFAULT_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
<|begin_of_text|>
{{ messages[0]['content'] }}
{% endif %}
{% for message in messages %}
{% if message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|>
{% endif %}"""

        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        print("Chat template set successfully!")

        # Set special tokens if needed
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token = "<|begin_of_text|>"
            tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(
                "<|begin_of_text|>")
            print(f"Set BOS token: {tokenizer.bos_token}")

        # Ensure assistant and user tokens are in the tokenizer's vocabulary
        special_tokens_dict = {}
        if "<|user|>" not in tokenizer.get_vocab():
            special_tokens_dict["additional_special_tokens"] = [
                "<|user|>", "<|assistant|>"]

        if special_tokens_dict:
            tokenizer.add_special_tokens(special_tokens_dict)
            print("Added special tokens to tokenizer vocabulary.")

    # Create OpenOrcaDataset instances
    # train_dataset = OpenOrcaDataset(
    #     train_dataset, tokenizer, max_length, include_labels)
    eval_dataset = OpenOrcaDataset(
        eval_dataset, tokenizer, max_length, include_labels)

    # Create DataLoaders
    train_dataloader = None
    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True)

    # For MPS, use pin_memory=False since MPS doesn't support pinned memory
    # and num_workers=0 to avoid potential issues with fork on macOS
    if device.type == "mps":
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=False, num_workers=0
        )
    else:
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=2
        )

    print("Dataset preparation complete!")
    return train_dataloader, eval_dataloader, tokenizer


def calculate_perplexity(model, eval_dataloader, device):
    """
    Calculate perplexity on the evaluation dataset.

    Args:
        model: Loaded language model
        eval_dataloader: DataLoader for evaluation
        device: Device to run inference on

    Returns:
        float: Perplexity score
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    print("Calculating perplexity...")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # For MPS, handle potential OOM errors
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Get loss and mask out padding tokens
                loss = outputs.loss

                # Calculate number of tokens (excluding padding)
                num_tokens = attention_mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

                # For MPS, explicitly clean up tensors to help with memory management
                if device.type == "mps":
                    del input_ids, attention_mask, labels, outputs, loss
                    # Force MPS to synchronize and release memory
                    torch.mps.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "MPS backend reached" in str(e):
                    print(f"Warning: {e}. Skipping this batch.")
                    # Clean up tensors that may be using memory
                    del input_ids, attention_mask, labels
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise e

    # Calculate perplexity
    if total_tokens == 0:
        print("Warning: No tokens processed, cannot calculate perplexity.")
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def generate_responses(model, eval_dataloader, tokenizer, device, generation_params):
    """
    Generate responses for the evaluation dataset.

    Args:
        model: Loaded language model
        eval_dataloader: DataLoader for evaluation
        tokenizer: Tokenizer for decoding outputs
        device: Device to run inference on
        generation_params: Dictionary of generation parameters

    Returns:
        list: Generated responses
        list: Ground truth responses
        float: Average generation time per sample
    """
    model.eval()
    responses = []
    ground_truths = []
    generation_times = []

    print("Generating responses...")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            original_responses = batch['original_response']

            # Time the generation
            start_time = time.time()

            try:
                # Generate responses
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_params
                )

                end_time = time.time()
                generation_time = (end_time - start_time) / \
                    input_ids.size(0)  # Time per sample
                generation_times.append(generation_time)

                # Extract generated text
                for i, output in enumerate(outputs):
                    # Get the generated text, skipping the input prompt
                    input_length = input_ids[i].size(0)
                    generated_ids = output[input_length:]

                    # Decode the generated text
                    generated_text = tokenizer.decode(
                        generated_ids, skip_special_tokens=True)
                    responses.append(generated_text)
                    ground_truths.append(original_responses[i])

                # For MPS, explicitly clean up tensors to help with memory management
                if device.type == "mps":
                    del input_ids, attention_mask, outputs
                    # Force MPS to synchronize and release memory
                    torch.mps.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "MPS backend reached" in str(e):
                    print(f"Warning: {e}. Skipping this batch.")
                    # Add empty responses to maintain batch alignment
                    for i in range(len(original_responses)):
                        responses.append("[Generation failed - out of memory]")
                        ground_truths.append(original_responses[i])
                    # Add a high generation time to signal the issue
                    # 60 seconds as a placeholder
                    generation_times.append(60.0)
                    # Clean up tensors that may be using memory
                    del input_ids, attention_mask
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise e

    if not generation_times:
        print("Warning: No responses were generated successfully.")
        return responses, ground_truths, 0.0

    avg_generation_time = mean(generation_times)
    return responses, ground_truths, avg_generation_time


def calculate_rouge_scores(generated_responses, ground_truth_responses):
    """
    Calculate ROUGE scores for generated responses.

    Args:
        generated_responses: List of generated responses
        ground_truth_responses: List of ground truth responses

    Returns:
        dict: ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for generated, ground_truth in zip(generated_responses, ground_truth_responses):
        scores = scorer.score(ground_truth, generated)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        'rouge1': mean(rouge1_scores),
        'rouge2': mean(rouge2_scores),
        'rougeL': mean(rougeL_scores)
    }


def evaluate_model(model_name, eval_batch_size=None, eval_sample_size=None, generation_configs=None):
    """
    Evaluate a model on OpenOrca dataset.

    Args:
        model_name: Name or path of the model to evaluate
        eval_batch_size: Batch size for evaluation (will use default for device if None)
        eval_sample_size: Number of samples to evaluate on (will use default for device if None)
        generation_configs: List of generation parameter dictionaries to try

    Returns:
        dict: Evaluation results
    """
    # Use device-specific defaults if not specified
    if eval_batch_size is None:
        eval_batch_size = DEFAULT_BATCH_SIZE
    if eval_sample_size is None:
        eval_sample_size = DEFAULT_SAMPLE_SIZE

    if generation_configs is None:
        generation_configs = [
            {
                "name": "default",
                "params": {
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            }
        ]

    results = {
        "model_name": model_name,
        "sample_size": eval_sample_size,
        "perplexity": None,
        "generation_results": []
    }

    print(f"\nEvaluating model: {model_name}")
    print(f"Sample size: {eval_sample_size}")
    print(f"Batch size: {eval_batch_size}")
    print(f"Device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    try:
        # First try loading with trust_remote_code, which might be needed for some models
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning when loading with trust_remote_code: {e}")
        # Fallback to loading without trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is set for any model
    if tokenizer.pad_token is None:
        print("Pad token not found, setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(
            f"Set pad token to: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # If the model name contains "llama" or "Llama", we might need to set a chat template
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower and (not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None):
        print("Llama model detected without chat template, setting a default one...")
        DEFAULT_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
<|begin_of_text|>
{{ messages[0]['content'] }}
{% endif %}
{% for message in messages %}
{% if message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|>
{% endif %}"""

        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        print("Set default Llama chat template")

        # Handle special tokens
        special_tokens_to_add = {}
        tokens_to_check = ["<|begin_of_text|>", "<|user|>", "<|assistant|>"]
        missing_tokens = []

        for token in tokens_to_check:
            if token not in tokenizer.get_vocab():
                missing_tokens.append(token)

        if missing_tokens:
            special_tokens_to_add["additional_special_tokens"] = missing_tokens
            tokenizer.add_special_tokens(special_tokens_to_add)
            print(f"Added missing special tokens: {missing_tokens}")

    try:
        # MPS compatible model loading
        if device.type == "mps":
            print("Loading model with MPS compatibility...")
            # For MPS, float16 is more compatible than bfloat16
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use float16 for MPS
                trust_remote_code=True
            ).to(device)
        else:
            # Standard loading for CUDA or CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
    except Exception as e:
        print(f"Warning when loading with optimized settings: {e}")
        # Fallback to basic loading with explicit device placement
        print("Falling back to basic model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(device)

    # Print model size info
    model_size = sum(p.numel()
                     for p in model.parameters()) / 1e6  # in millions
    print(f"Model loaded with {model_size:.2f}M parameters")

    # Build evaluation dataset
    print("Building evaluation dataset...")
    _, eval_dataloader, _ = build_openorca_dataset(
        sample_size=eval_sample_size,
        max_length=DEFAULT_MAX_LENGTH,
        batch_size=eval_batch_size,
        include_labels=True
    )

    # Calculate perplexity
    try:
        perplexity = calculate_perplexity(model, eval_dataloader, device)
        results["perplexity"] = perplexity
        print(f"Perplexity: {perplexity:.4f}")
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        print("Skipping perplexity calculation.")
        results["perplexity"] = "Error"

    # Build evaluation dataset without labels for generation (to use just the prompt)
    print("Building dataset for generation evaluation...")
    _, eval_gen_dataloader, tokenizer = build_openorca_dataset(
        sample_size=eval_sample_size,
        max_length=DEFAULT_MAX_LENGTH,
        batch_size=eval_batch_size,
        include_labels=False
    )

    # Evaluate generation with different configurations
    for config in generation_configs:
        config_name = config["name"]
        params = config["params"]

        print(f"\nEvaluating generation with configuration: {config_name}")
        try:
            responses, ground_truths, avg_time = generate_responses(
                model, eval_gen_dataloader, tokenizer, device, params
            )

            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(responses, ground_truths)

            tokens_per_second = params["max_new_tokens"] / \
                avg_time if avg_time > 0 else 0

            # Store results
            config_results = {
                "config_name": config_name,
                "generation_params": params,
                "rouge_scores": rouge_scores,
                "avg_generation_time": avg_time,
                "tokens_per_second": tokens_per_second,
                "sample_responses": [(resp, ground) for resp, ground in zip(responses[:3], ground_truths[:3])]
            }

            results["generation_results"].append(config_results)

            # Print results
            print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
            print(f"Avg generation time: {avg_time:.4f} seconds per sample")
            print(f"Tokens per second: {tokens_per_second:.2f}")
        except Exception as e:
            print(f"Error during generation with config {config_name}: {e}")
            print("Skipping this configuration.")
            config_results = {
                "config_name": config_name,
                "generation_params": params,
                "error": str(e)
            }
            results["generation_results"].append(config_results)

    return results


def main():
    # Define model to evaluate

    # Define generation configurations to test
    generation_configs = [
        {
            "name": "greedy",
            "params": {
                "max_new_tokens": 256,
                "do_sample": False,
            }
        },
        {
            "name": "sampling_temp0.7",
            "params": {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
    ]

    # Use simpler config for MPS to reduce memory usage
    if device.type == "mps":
        print("Using simplified generation configs for MPS device")
        generation_configs = [
            {
                "name": "greedy",
                "params": {
                    "max_new_tokens": 128,  # Reduced token count for MPS
                    "do_sample": False,
                }
            }
        ]

    # Evaluate model using device-specific defaults
    results = evaluate_model(
        model_name=MODEL,
        # Use defaults which are appropriate for the device
        eval_batch_size=None,
        eval_sample_size=None,
        generation_configs=generation_configs
    )

    # Create a timestamped file name for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device_suffix = f"_{device.type}"
    model = MODEL.split("/")[-1]
    results_file = f"eval_results_{model}_openorca{device_suffix}_{timestamp}.txt"

    # Open file for writing
    with open(results_file, "w") as f:
        # Write summary of results to file
        f.write("===== EVALUATION SUMMARY =====\n")
        f.write(f"Model: {results['model_name']}\n")
        f.write(f"Sample size: {results['sample_size']}\n")
        f.write(f"Device: {device}\n")
        if isinstance(results['perplexity'], (int, float)):
            f.write(f"Perplexity: {results['perplexity']:.4f}\n")
        else:
            f.write(f"Perplexity: {results['perplexity']}\n")

        f.write("\nGeneration Results:\n")
        for gen_result in results["generation_results"]:
            f.write(f"\nConfiguration: {gen_result['config_name']}\n")
            if 'error' in gen_result:
                f.write(f"Error: {gen_result['error']}\n")
                continue

            f.write(f"ROUGE-1: {gen_result['rouge_scores']['rouge1']:.4f}\n")
            f.write(f"ROUGE-2: {gen_result['rouge_scores']['rouge2']:.4f}\n")
            f.write(f"ROUGE-L: {gen_result['rouge_scores']['rougeL']:.4f}\n")
            f.write(
                f"Avg generation time: {gen_result['avg_generation_time']:.4f} seconds\n")
            f.write(
                f"Tokens per second: {gen_result['tokens_per_second']:.2f}\n")

        f.write("\n===== SAMPLE RESPONSES =====\n")
        # Check if we have successful generations before trying to print samples
        successful_generations = [
            g for g in results["generation_results"] if 'sample_responses' in g]
        if successful_generations:
            first_successful = successful_generations[0]
            for i, (resp, ground) in enumerate(first_successful["sample_responses"]):
                f.write(f"\nSample {i+1}:\n")
                f.write(f"Generated: {resp[:150]}...\n")
                f.write(f"Ground truth: {ground[:150]}...\n")
        else:
            f.write("No successful generations to show samples for.\n")

        f.write("\nEvaluation complete!\n")

    # Print summary of results to console
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Model: {results['model_name']}")
    print(f"Sample size: {results['sample_size']}")
    print(f"Device: {device}")
    if isinstance(results['perplexity'], (int, float)):
        print(f"Perplexity: {results['perplexity']:.4f}")
    else:
        print(f"Perplexity: {results['perplexity']}")

    print("\nGeneration Results:")
    for gen_result in results["generation_results"]:
        print(f"\nConfiguration: {gen_result['config_name']}")
        if 'error' in gen_result:
            print(f"Error: {gen_result['error']}")
            continue

        print(f"ROUGE-1: {gen_result['rouge_scores']['rouge1']:.4f}")
        print(f"ROUGE-2: {gen_result['rouge_scores']['rouge2']:.4f}")
        print(f"ROUGE-L: {gen_result['rouge_scores']['rougeL']:.4f}")
        print(
            f"Avg generation time: {gen_result['avg_generation_time']:.4f} seconds")
        print(f"Tokens per second: {gen_result['tokens_per_second']:.2f}")

    print("\n===== SAMPLE RESPONSES =====")
    # Check if we have successful generations before trying to print samples
    successful_generations = [
        g for g in results["generation_results"] if 'sample_responses' in g]
    if successful_generations:
        first_successful = successful_generations[0]
        for i, (resp, ground) in enumerate(first_successful["sample_responses"]):
            print(f"\nSample {i+1}:")
            print(f"Generated: {resp[:150]}...")
            print(f"Ground truth: {ground[:150]}...")
    else:
        print("No successful generations to show samples for.")

    print(f"\nEvaluation complete! Results saved to {results_file}")


if __name__ == "__main__":
    main()
