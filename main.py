# 1.  ───── Install deps (PyTorch assumed) ─────────────────────────

# 2.  ───── Python script (eval_glaive_qwen3.py) ───────────────────
import re, json, argparse, ast, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=2_000,   # full = 113 k
                    help="How many rows to evaluate (set -1 for full set)")
parser.add_argument("--max_new",   type=int, default=192)
args = parser.parse_args()

# --- 2.1 Load model/tokenizer -------------------------------------------------
tok   = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",
                                      trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype="auto",
            device_map="auto")

# --- 2.2 Load dataset ---------------------------------------------------------
ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")  # 113 k rows&#8203;:contentReference[oaicite:0]{index=0}
if args.n_samples > 0:
    ds = ds.shuffle(seed=42).select(range(args.n_samples))

# Helper: pull out (messages, gold_call_json)
def parse_row(row):
    # Each row contains two strings: 'system' & 'chat'
    system_msg = row["system"]
    chat       = row["chat"]
    # We split by <|endoftext|> to get turns
    turns = [t.strip() for t in chat.split("<|endoftext|>") if t.strip()]
    msgs  = [{"role": "system", "content": system_msg}]
    gold_call = None
    for t in turns:
        if t.startswith("USER:"):
            msgs.append({"role": "user",
                         "content": t.removeprefix("USER:").strip()})
        elif "ASSISTANT:" in t and "<functioncall>" not in t:
            msgs.append({"role": "assistant",
                         "content": t.split("ASSISTANT:")[1].strip()})
        elif "<functioncall>" in t:           # gold function call
            content = t.split("<functioncall>")[1].strip().replace('\n', '')
            print(content)
            first_quote_pos = content.find("arguments") + len("arguments") + content[content.find("arguments") + len("arguments"):].find("'")
            last_quote_pos = last_quote_pos = content.rfind("'")
            content_list = list(content)
            content_list[first_quote_pos] = ""
            content_list[last_quote_pos] = ""
            content = "".join(content_list)
            content = content.replace("true", "'True'")
            content = content.replace("false", "'False'")
            content = ast.literal_eval(content)
            arguments = content["arguments"]
            if type(arguments) == str:
                arguments = arguments.replace("true", "'True'")
                arguments = arguments.replace("false", "'False'")
                arguments = ast.literal_eval(arguments)
                
            content["arguments"] = arguments
            content = json.dumps(content)
            gold_call = json.loads(re.search(r"\{.*\}", content, re.S).group())
            break                              # we stop right before function call
    return msgs, gold_call

# --- 2.3 Metric accumulators --------------------------------------------------
json_valid, name_match, exact_match = [], [], []

# --- 2.4 Iterate --------------------------------------------------------------
for sample in ds:
    messages, gold = parse_row(sample)
    prompt = tok.apply_chat_template(messages,
                                     tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(**inputs,
                         max_new_tokens=args.max_new,
                         do_sample=False)
    gen = tok.decode(out[0][inputs.input_ids.shape[-1]:],
                     skip_special_tokens=False)

    # Extract model’s function call (basic regex)
    m = re.search(r"<functioncall>\s*(\{.*?\})", gen, re.S)
    try:
        pred_call = json.loads(m.group(1)) if m else None
        json_valid.append(1)
    except Exception:
        pred_call = None
        json_valid.append(0)

    # Compare
    if pred_call and gold:
        name_match.append(int(pred_call.get("name") == gold["name"]))
        exact_match.append(int(pred_call == gold))
    else:
        name_match.append(0)
        exact_match.append(0)

# --- 2.5 Report ---------------------------------------------------------------
def pct(lst): return 100 * np.mean(lst)
print(f"\nJSON-valid %          : {pct(json_valid):5.1f}")
print(f"Name-match %          : {pct(name_match):5.1f}")
print(f"Exact-match % (strict): {pct(exact_match):5.1f}")
