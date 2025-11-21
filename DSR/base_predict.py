#!/usr/bin/env python3

import json
import os
from typing import List, Dict

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


TEST_FILE = ""
MODEL_DIR = ""
PRETRAINED_DIR = ""
OUT_FILE = ""

BATCH_SIZE = None
MAX_LENGTH = None
MAX_NEW_TOKENS = None
MIN_NEW_TOKENS = None
TEMPERATURE = None
TOP_P = None
DO_SAMPLE = None

PROMPT_TEMPLATE = (
    "Task: Reconstruct complete sentence from segmented translations.\n"
    "Sample ID: {sample_id}\n"
    "Known segments:\n"
    "{segments}\n"
    "Please output a natural, coherent, and semantically complete sentence."
)


def load_test(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fr:
        return [json.loads(line) for line in fr if line.strip()]


def build_conversation(rec: Dict) -> List[Dict]:
    segs = []
    idx = 1
    while True:
        key = f"seg{idx}"
        if key not in rec:
            break
        if rec[key]:
            segs.append(f"{idx}) {rec[key]}")
        idx += 1
    segment_block = "\n".join(segs) if segs else ""
    user_prompt = PROMPT_TEMPLATE.format(sample_id=rec.get("id", ""), segments=segment_block)
    return [
        {"role": "user", "content": user_prompt},
    ]


def main() -> None:
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.padding_side = 'left'
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    try:
        from safetensors.torch import load_file as safe_load
        weight_path = None
        if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
            weight_path = os.path.join(MODEL_DIR, "model.safetensors")
        elif os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
            weight_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
        state_dict = None
        if weight_path and weight_path.endswith(".safetensors"):
            state_dict = safe_load(weight_path)
        elif weight_path and weight_path.endswith(".bin"):
            state_dict = torch.load(weight_path, map_location="cpu")
        else:
            shard_states = {}
            for fname in os.listdir(MODEL_DIR):
                if fname.endswith(".safetensors"):
                    shard = safe_load(os.path.join(MODEL_DIR, fname))
                    shard_states.update(shard)
            if shard_states:
                state_dict = shard_states
        if state_dict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except Exception:
        pass
    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass
    try:
        model.generation_config.use_cache = False
    except Exception:
        pass
    model.eval()

    data = load_test(TEST_FILE)
    with open(OUT_FILE, "w", encoding="utf-8") as fw:
        for i in tqdm(range(0, len(data), BATCH_SIZE if BATCH_SIZE else 1), desc="Predicting", unit="batch"):
            batch = data[i:i+(BATCH_SIZE if BATCH_SIZE else 1)]
            convs = [build_conversation(rec) for rec in batch]
            prompt_texts = tokenizer.apply_chat_template(
                convs,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH if MAX_LENGTH else 512,
                return_tensors='pt',
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS if MAX_NEW_TOKENS else None,
                    min_new_tokens=MIN_NEW_TOKENS if MIN_NEW_TOKENS else None,
                    do_sample=DO_SAMPLE if DO_SAMPLE is not None else True,
                    temperature=TEMPERATURE if TEMPERATURE is not None else None,
                    top_p=TOP_P if TOP_P is not None else None,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=False,
                )
            gen_texts = tokenizer.batch_decode(gen_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

            for rec, pred in zip(batch, gen_texts):
                out = {
                    "id": rec.get("id", ""),
                    "pred": pred.strip(),
                    "ori": rec.get("ori", ""),
                }
                idx2 = 1
                while True:
                    key = f"seg{idx2}"
                    if key not in rec:
                        break
                    out[key] = rec[key]
                    idx2 += 1
                fw.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Predictions written to: {OUT_FILE}")


if __name__ == "__main__":
    main()
