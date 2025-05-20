import os
import torch
import time
import json
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from statistics import mean
from datetime import datetime

# Check for MPS (Apple Silicon GPU) availability
MODEL = "meta-llama/Llama-3.2-3B-Instruct"


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
    DEFAULT_MAX_LENGTH = 1024
    DEFAULT_SAMPLE_SIZE = 50
    print(f"Configured for MPS device with optimized memory settings")
    # Set environment variable to control MPS memory usage
    # Use minimal preallocated memory
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
else:
    # Regular settings for CUDA or CPU
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_MAX_LENGTH = 1024
    DEFAULT_SAMPLE_SIZE = 100

# Define a custom dataset class for Berkeley Function Calling Leaderboard dataset


class BerkeleyFunctionCallingDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=1024, include_labels=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels

        # Debug the first item to understand the dataset structure
        if len(dataset) > 0:
            print("\nDataset item structure example:")
            first_item = dataset[0]
            for key in first_item.keys():
                print(f"Key: {key}, Type: {type(first_item[key])}")
            print()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get item from dataset
        item = self.dataset[idx]

        # Extract question (chat messages) with improved handling for different formats
        messages = []
        question_data = item.get("question", None)

        # Handle different possible formats more robustly
        if question_data is not None:
            # Format 1: Direct message object
            if isinstance(question_data, dict) and "role" in question_data and "content" in question_data:
                messages = [question_data]

            # Format 2: List of messages
            elif isinstance(question_data, list):
                if len(question_data) > 0:
                    # Format 2a: List of message objects
                    if isinstance(question_data[0], dict) and "role" in question_data[0]:
                        messages = question_data

                    # Format 2b: List of lists of message objects
                    elif isinstance(question_data[0], list) and len(question_data[0]) > 0:
                        messages = question_data[0]

            # Fallback for raw text: create a user message
            elif isinstance(question_data, str) and question_data.strip():
                messages = [{"role": "user", "content": question_data}]

        # Fallback if nothing worked
        if not messages:
            # Check if there's a 'prompt' or 'instruction' field we can use
            for field in ["prompt", "instruction", "input"]:
                text = item.get(field, "")
                if isinstance(text, str) and text.strip():
                    messages = [{"role": "user", "content": text}]
                    break

        # Final fallback
        if not messages:
            messages = [
                {"role": "user", "content": "Provide function calling assistance"}]

        # Extract function definitions with better error handling
        functions = []
        function_data = item.get("function", None)

        if function_data is not None:
            if isinstance(function_data, list):
                functions = function_data
            elif isinstance(function_data, dict):
                # Single function as dict
                functions = [function_data]
            elif isinstance(function_data, str) and function_data.strip():
                # Try to parse JSON string
                try:
                    parsed = json.loads(function_data)
                    if isinstance(parsed, list):
                        functions = parsed
                    elif isinstance(parsed, dict):
                        functions = [parsed]
                except json.JSONDecodeError:
                    # Not valid JSON, use as description
                    functions = [{"name": "function",
                                  "description": function_data}]

        # Extract ground truth (expected function call)
        ground_truth = ""

        # Try different possible fields for ground truth
        for field in ["ground_truth", "answer", "response", "output"]:
            gt_data = item.get(field, None)
            if gt_data:
                if isinstance(gt_data, list) and len(gt_data) > 0:
                    # Take first item if it's a list
                    if isinstance(gt_data[0], str):
                        ground_truth = gt_data[0]
                    elif isinstance(gt_data[0], dict) and "content" in gt_data[0]:
                        ground_truth = gt_data[0]["content"]
                elif isinstance(gt_data, str):
                    ground_truth = gt_data
                elif isinstance(gt_data, dict) and "content" in gt_data:
                    ground_truth = gt_data["content"]

                if ground_truth:
                    break

        # Convert function list to a formatted string for the prompt
        function_str = json.dumps(functions, indent=2) if functions else "{}"

        # Create a formatted system prompt with the functions
        system_content = f"You are a helpful AI assistant that can use functions. Here are the available functions:\n\n{function_str}\n\nTo use a function, respond with the function name and parameters in this format: function_name(param1=value1, param2=value2)"

        # Check if there's already a system message in the messages list
        has_system_message = any(
            msg.get("role") == "system" for msg in messages)

        # Prepare chat messages for the model
        chat = []

        # Add system message if not already present
        if not has_system_message:
            chat.append({"role": "system", "content": system_content})

        # Add all messages from the question
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                chat.append(msg)
            elif isinstance(msg, str):
                # If it's just a string, treat as user message
                chat.append({"role": "user", "content": msg})

        # Convert chat to tokenizer format
        try:
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template application failed with error: {e}")
            print("Falling back to manual formatting...")
            # Manual fallback formatting
            prompt_parts = []

            # Add system message first if not already in messages
            if not has_system_message:
                prompt_parts.append(f"System: {system_content}")

            # Add all other messages
            for msg in chat:
                if msg["role"] != "system":  # Skip system since we handled it above
                    role = msg.get("role", "").capitalize()
                    content = msg.get("content", "")
                    prompt_parts.append(f"{role}: {content}")

            prompt = "\n".join(prompt_parts)

            # If we're not including labels and last message isn't from assistant,
            # add an assistant prompt to generate a response
            if self.include_labels is False and (not messages or messages[-1].get("role") != "assistant"):
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
            'original_question': json.dumps(messages),
            'original_functions': function_str,
            'original_ground_truth': ground_truth,
            'item_id': item.get("id", f"item_{idx}")
        }

        if self.include_labels:
            # For evaluation, we need to tokenize the ground truth
            # This is the expected function call
            ground_truth_encodings = self.tokenizer(
                ground_truth, max_length=self.max_length, padding="max_length", truncation=True)
            ground_truth_ids = torch.tensor(
                ground_truth_encodings['input_ids'])

            # For perplexity calculation on function calls, we need labels
            # Create labels as -100 (ignore index) except for the ground truth part
            result['labels'] = torch.full_like(input_ids, -100)

            # For BFCL, we really only care about exact function call match, so we'll
            # use the entire input_ids as labels for simplicity
            # A more sophisticated approach would target only the function call portion
            result['labels'] = input_ids.clone()
            result['labels'][result['labels'] ==
                             self.tokenizer.pad_token_id] = -100

        return result


def build_berkeley_dataset(sample_size=None, max_length=None, batch_size=None, include_labels=True):
    """
    Build a dataset from Berkeley Function Calling Leaderboard for evaluating function calling.

    Args:
        sample_size: Number of examples to include in the dataset (uses DEFAULT_SAMPLE_SIZE if None)
        max_length: Maximum sequence length for tokenization (uses DEFAULT_MAX_LENGTH if None)
        batch_size: Batch size for dataloader (uses DEFAULT_BATCH_SIZE if None)
        include_labels: Whether to include labels in the dataset (for training or perplexity)

    Returns:
        train_dataloader: DataLoader for training data (or None if not needed)
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
        f"Building Berkeley dataset with sample_size={sample_size}, max_length={max_length}, batch_size={batch_size}")
    print("Loading Berkeley Function Calling Leaderboard dataset...")

    dataset = None

    # Option 1: Try loading directly from HF hub
    try:
        print("Trying to load from Hugging Face hub...")
        dataset = load_dataset(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard", streaming=True)
        print("Dataset loaded successfully from Hugging Face hub!")

        # Get the train split if it exists
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        dataset = list(dataset[split_name].take(sample_size))
    except Exception as e:
        print(f"Error loading dataset from Hugging Face hub: {e}")
        dataset = None

    # Option 2: Try loading from local files if HF load failed
    if dataset is None:
        try:
            print(
                "Trying to load from local Berkeley-Function-Calling-Leaderboard directory...")
            # Look for JSON files in the repository directory
            dataset = load_dataset("json",
                                   data_files={
                                       "train": "Berkeley-Function-Calling-Leaderboard/**/*.json"},
                                   streaming=True)
            print("Dataset loaded from local files successfully!")

            # Convert streaming dataset to list
            dataset = list(dataset["train"].take(sample_size))
        except Exception as e2:
            print(f"Error loading from local files: {e2}")

            # Option 3: Try direct file loading as fallback
            try:
                print("Trying to directly load a specific JSON file...")
                specific_file = "Berkeley-Function-Calling-Leaderboard/BFCL_v3_exec_multiple.json"
                if os.path.exists(specific_file):
                    dataset = load_dataset("json", data_files={
                                           "train": specific_file})
                    print(f"Successfully loaded {specific_file}!")
                    dataset = list(dataset["train"].take(sample_size))
                else:
                    print(f"File {specific_file} not found.")
                    # Last resort: Search for any JSON file
                    import glob
                    json_files = glob.glob(
                        "Berkeley-Function-Calling-Leaderboard/**/*.json", recursive=True)
                    if json_files:
                        print(f"Found JSON files: {json_files[0]}")
                        dataset = load_dataset("json", data_files={
                                               "train": json_files[0]})
                        dataset = list(dataset["train"].take(sample_size))
                    else:
                        print(
                            "No JSON files found in the Berkeley-Function-Calling-Leaderboard directory.")
                        raise FileNotFoundError(
                            "No Berkeley dataset files found")
            except Exception as e3:
                print(f"All attempts to load the dataset failed: {e3}")
                raise Exception(
                    "Failed to load the Berkeley Function Calling dataset after all attempts")

    if not dataset:
        raise ValueError(
            "Dataset loading failed. Please ensure the Berkeley-Function-Calling-Leaderboard data is available.")

    # Print a sample data item for debugging
    if len(dataset) > 0:
        print("\nSample data item structure:")
        sample_keys = dataset[0].keys()
        print(f"Keys: {', '.join(sample_keys)}")

        # Check for expected keys
        expected_keys = ["question", "function", "ground_truth"]
        missing_keys = [key for key in expected_keys if key not in sample_keys]
        if missing_keys:
            print(
                f"Warning: Expected keys missing from dataset: {missing_keys}")
            print("Will attempt to adapt to the available structure.")

        # Show structure preview
        for key in sample_keys:
            value = dataset[0][key]
            print(f"\nKey: {key}")
            print(f"Type: {type(value)}")
            if isinstance(value, list) and len(value) > 0:
                print(f"First element type: {type(value[0])}")
                print(f"List length: {len(value)}")
                if len(value) > 0 and isinstance(value[0], dict) and len(value[0]) > 0:
                    print(f"First element keys: {list(value[0].keys())}")

    # Split dataset - use all for evaluation in this case
    eval_dataset = dataset
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Load Llama tokenizer
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
        # Default Llama-3 chat template with added tools/functions support
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

    # Create BerkeleyFunctionCallingDataset instance
    eval_dataset = BerkeleyFunctionCallingDataset(
        eval_dataset, tokenizer, max_length, include_labels)

    # Create DataLoaders
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
    return None, eval_dataloader, tokenizer


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
    batch_count = 0
    success_count = 0

    print("Calculating perplexity...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
            try:
                # Debug first batch
                if batch_idx == 0:
                    print(f"\nFirst batch info:")
                    print(f"input_ids shape: {batch['input_ids'].shape}")
                    print(
                        f"attention_mask shape: {batch['attention_mask'].shape}")
                    if 'labels' in batch:
                        print(f"labels shape: {batch['labels'].shape}")
                        # Check if labels contain valid indices
                        unique_labels = torch.unique(batch['labels']).tolist()
                        print(f"Unique label values: {unique_labels[:10]}..." if len(
                            unique_labels) > 10 else f"Unique label values: {unique_labels}")
                        if -100 not in unique_labels:
                            print(
                                "Warning: No -100 values in labels. This might cause issues with masked loss calculation.")
                    else:
                        print(
                            "Warning: 'labels' not found in batch. This will cause perplexity calculation to fail.")

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(
                    device) if 'labels' in batch else None

                if labels is None:
                    print(
                        f"Skipping batch {batch_idx}: No labels available for perplexity calculation")
                    continue

                # For MPS, handle potential OOM errors
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Get loss and mask out padding tokens
                loss = outputs.loss

                # Calculate number of tokens (excluding padding)
                num_tokens = attention_mask.sum().item()

                # Skip batches with invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping batch {batch_idx}: Loss is {loss.item()}")
                    continue

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                success_count += 1

                # For MPS, explicitly clean up tensors to help with memory management
                if device.type == "mps":
                    del input_ids, attention_mask, labels, outputs, loss
                    # Force MPS to synchronize and release memory
                    torch.mps.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "MPS backend reached" in str(e):
                    print(f"Warning: {e}. Skipping batch {batch_idx}.")
                    # Clean up tensors that may be using memory
                    if 'input_ids' in locals():
                        del input_ids
                    if 'attention_mask' in locals():
                        del attention_mask
                    if 'labels' in locals():
                        del labels
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(f"Error in batch {batch_idx}: {e}")
                    raise e
            except Exception as e:
                print(f"Unexpected error in batch {batch_idx}: {e}")
                continue

            batch_count += 1

    # Calculate perplexity
    if total_tokens == 0:
        print("Warning: No tokens processed, cannot calculate perplexity.")
        return float('inf')

    print(
        f"Processed {success_count} successful batches out of {batch_count} total batches")
    print(f"Total tokens processed: {total_tokens}")

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
        list: Item IDs 
        float: Average generation time per sample
    """
    model.eval()
    responses = []
    ground_truths = []
    item_ids = []
    generation_times = []
    batch_count = 0
    success_count = 0

    print("Generating responses...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
            # Debug the first batch
            if batch_idx == 0:
                print(f"\nFirst batch info:")
                print(f"input_ids shape: {batch['input_ids'].shape}")
                print(f"attention_mask shape: {batch['attention_mask'].shape}")

                # Preview ground truth format
                orig_gt = batch['original_ground_truth']
                print(f"Ground truth sample: '{orig_gt[0]}'")

                # Preview the input to understand what we're generating from
                input_text = tokenizer.decode(
                    batch['input_ids'][0], skip_special_tokens=True)
                print(
                    f"Input text sample (truncated): '{input_text[:100]}...'")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            original_ground_truths = batch['original_ground_truth']
            batch_item_ids = batch['item_id']

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
                success_count += 1

                # Extract generated text
                for i, output in enumerate(outputs):
                    # Get the generated text, skipping the input prompt
                    input_length = input_ids[i].size(0)
                    generated_ids = output[input_length:]

                    # Decode the generated text
                    generated_text = tokenizer.decode(
                        generated_ids, skip_special_tokens=True)

                    # Clean up the generated text - we're only interested in the function call
                    generated_text = generated_text.strip()

                    # Extract function call using regex with improved pattern
                    # This pattern matches function_name(param1=value1, param2="value2", param3=3)
                    function_call_match = re.search(
                        r'(\w+)\s*\((.*?)\)', generated_text)

                    if function_call_match:
                        # Extract function name and arguments
                        func_name = function_call_match.group(1)
                        args_str = function_call_match.group(2)
                        # Recreate the cleaned function call
                        cleaned_function_call = f"{func_name}({args_str})"
                        generated_text = cleaned_function_call
                    else:
                        # If we can't find a function call using regex, check for JSON format
                        try:
                            # Try to find a JSON object in the text
                            json_match = re.search(
                                r'(\{.*?\})', generated_text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                                # Try to parse as JSON to validate
                                json_obj = json.loads(json_str)
                                # If it has name and arguments fields, it might be a function call
                                if "name" in json_obj and "arguments" in json_obj:
                                    func_name = json_obj["name"]
                                    args = json_obj["arguments"]
                                    # Format as function call
                                    if isinstance(args, dict):
                                        args_str = ", ".join(
                                            f"{k}={json.dumps(v)}" for k, v in args.items())
                                        generated_text = f"{func_name}({args_str})"
                                    else:
                                        generated_text = f"{func_name}({args})"
                        except (json.JSONDecodeError, ValueError):
                            # If JSON parsing fails, keep the original text
                            pass

                    # If first batch, print some examples for debugging
                    if batch_idx == 0 and i < 2:
                        print(f"\nGeneration example {i+1}:")
                        print(f"Raw output: '{generated_text}'")
                        print(f"Ground truth: '{original_ground_truths[i]}'")

                    responses.append(generated_text)
                    ground_truths.append(original_ground_truths[i])
                    item_ids.append(batch_item_ids[i])

                # For MPS, explicitly clean up tensors to help with memory management
                if device.type == "mps":
                    del input_ids, attention_mask, outputs
                    # Force MPS to synchronize and release memory
                    torch.mps.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "MPS backend reached" in str(e):
                    print(f"Warning: {e}. Skipping batch {batch_idx}.")
                    # Add empty responses to maintain batch alignment
                    for i in range(len(original_ground_truths)):
                        responses.append("[Generation failed - out of memory]")
                        ground_truths.append(original_ground_truths[i])
                        item_ids.append(batch_item_ids[i])
                    # Add a high generation time to signal the issue
                    # 60 seconds as a placeholder
                    generation_times.append(60.0)
                    # Clean up tensors that may be using memory
                    if 'input_ids' in locals():
                        del input_ids
                    if 'attention_mask' in locals():
                        del attention_mask
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(f"Error in batch {batch_idx}: {e}")
                    raise e
            except Exception as e:
                print(f"Unexpected error in batch {batch_idx}: {e}")
                # Add error message as response
                for i in range(len(original_ground_truths)):
                    responses.append(f"[Generation error: {str(e)}]")
                    ground_truths.append(original_ground_truths[i])
                    item_ids.append(batch_item_ids[i])

            batch_count += 1

    if not generation_times:
        print("Warning: No responses were generated successfully.")
        return responses, ground_truths, item_ids, 0.0

    print(
        f"Processed {success_count} successful batches out of {batch_count} total batches")
    print(f"Generated {len(responses)} responses")

    avg_generation_time = mean(generation_times)
    return responses, ground_truths, item_ids, avg_generation_time


def calculate_function_call_accuracy(generated_responses, ground_truth_responses):
    """
    Calculate exact match and partial match rates for function calls.

    Args:
        generated_responses: List of generated function calls
        ground_truth_responses: List of ground truth function calls

    Returns:
        dict: Accuracy metrics
    """
    exact_matches = 0
    function_name_matches = 0
    total = len(generated_responses)

    for generated, ground_truth in zip(generated_responses, ground_truth_responses):
        # Trim whitespace
        generated = generated.strip()
        ground_truth = ground_truth.strip()

        # Check for exact match (case sensitive)
        if generated == ground_truth:
            exact_matches += 1
            function_name_matches += 1
            continue

        # Extract function name from generated and ground truth
        gen_func_match = re.search(r'(\w+)\s*\(', generated)
        gt_func_match = re.search(r'(\w+)\s*\(', ground_truth)

        if gen_func_match and gt_func_match:
            gen_func_name = gen_func_match.group(1)
            gt_func_name = gt_func_match.group(1)

            # Check if the function names match
            if gen_func_name.lower() == gt_func_name.lower():
                function_name_matches += 1

    exact_match_rate = exact_matches / total if total > 0 else 0
    function_name_match_rate = function_name_matches / total if total > 0 else 0

    return {
        "exact_match": exact_match_rate,
        "function_name_match": function_name_match_rate
    }


def evaluate_model(model_name, eval_batch_size=None, eval_sample_size=None, generation_configs=None):
    """
    Evaluate a model on Berkeley Function Calling Leaderboard dataset.

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
                "name": "greedy",
                "params": {
                    "max_new_tokens": 256,
                    "do_sample": False,
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
    try:
        _, eval_dataloader, _ = build_berkeley_dataset(
            sample_size=eval_sample_size,
            max_length=DEFAULT_MAX_LENGTH,
            batch_size=eval_batch_size,
            include_labels=True
        )
    except Exception as e:
        print(f"Error building evaluation dataset: {e}")
        raise

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
    _, eval_gen_dataloader, tokenizer = build_berkeley_dataset(
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
            responses, ground_truths, item_ids, avg_time = generate_responses(
                model, eval_gen_dataloader, tokenizer, device, params
            )

            # Calculate function call accuracy
            accuracy_metrics = calculate_function_call_accuracy(
                responses, ground_truths)

            tokens_per_second = params["max_new_tokens"] / \
                avg_time if avg_time > 0 else 0

            # Store results
            config_results = {
                "config_name": config_name,
                "generation_params": params,
                "accuracy": accuracy_metrics,
                "avg_generation_time": avg_time,
                "tokens_per_second": tokens_per_second,
                "sample_responses": [(resp, ground, item_id) for resp, ground, item_id in zip(responses[:5], ground_truths[:5], item_ids[:5])]
            }

            results["generation_results"].append(config_results)

            # Print results
            print(
                f"Exact Match Accuracy: {accuracy_metrics['exact_match']:.4f}")
            print(
                f"Function Name Match Accuracy: {accuracy_metrics['function_name_match']:.4f}")
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
    model = MODEL.split('/')[-1]
    results_file = f"eval_results_{model}_bfcl{device_suffix}_{timestamp}.txt"

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

            f.write(
                f"Exact Match Accuracy: {gen_result['accuracy']['exact_match']:.4f}\n")
            f.write(
                f"Function Name Match Accuracy: {gen_result['accuracy']['function_name_match']:.4f}\n")
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
            for i, (resp, ground, item_id) in enumerate(first_successful["sample_responses"]):
                f.write(f"\nSample {i+1} (ID: {item_id}):\n")
                f.write(f"Generated: {resp}\n")
                f.write(f"Ground truth: {ground}\n")
                match = "✓ EXACT MATCH" if resp.strip() == ground.strip() else "✗ NO MATCH"
                f.write(f"Result: {match}\n")
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

        print(
            f"Exact Match Accuracy: {gen_result['accuracy']['exact_match']:.4f}")
        print(
            f"Function Name Match Accuracy: {gen_result['accuracy']['function_name_match']:.4f}")
        print(
            f"Avg generation time: {gen_result['avg_generation_time']:.4f} seconds")
        print(f"Tokens per second: {gen_result['tokens_per_second']:.2f}")

    print("\n===== SAMPLE RESPONSES =====")
    # Check if we have successful generations before trying to print samples
    successful_generations = [
        g for g in results["generation_results"] if 'sample_responses' in g]
    if successful_generations:
        first_successful = successful_generations[0]
        for i, (resp, ground, item_id) in enumerate(first_successful["sample_responses"]):
            print(f"\nSample {i+1} (ID: {item_id}):")
            print(f"Generated: {resp}")
            print(f"Ground truth: {ground}")
            match = "✓ EXACT MATCH" if resp.strip() == ground.strip() else "✗ NO MATCH"
            print(f"Result: {match}")
    else:
        print("No successful generations to show samples for.")

    print(f"\nEvaluation complete! Results saved to {results_file}")


if __name__ == "__main__":
    main()
