#!/usr/bin/env python3

import json
import os
from typing import List, Dict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from dsr import DSRModule


TEST_FILE = ""
MODEL_DIR = ""
PRETRAINED_DIR = ""
OUT_FILE = ""

STEPS = None
GEN_LENGTH = None
BLOCK_LENGTH = None
TEMPERATURE = None
CFG_SCALE = None
LANGUAGE = None
USE_LEXMASKER = None


def load_test_data(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_segments(rec: Dict) -> List[str]:
    segments = []
    idx = 1
    while True:
        key = f"seg{idx}"
        if key not in rec:
            break
        if rec.get(key):
            segments.append(rec[key])
        idx += 1
    return segments


def main() -> None:
    os.makedirs(os.path.dirname(OUT_FILE) if OUT_FILE else '.', exist_ok=True)
    
    dsr = DSRModule(
        model_path=PRETRAINED_DIR,
        tokenizer_path=PRETRAINED_DIR,
        language=LANGUAGE if LANGUAGE else 'de',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_lexmasker=USE_LEXMASKER if USE_LEXMASKER is not None else True,
        preserve_named_entities=True,
        preserve_numerals=True
    )
    
    if MODEL_DIR and os.path.exists(MODEL_DIR):
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
                missing, unexpected = dsr.model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass
    
    data = load_test_data(TEST_FILE)
    
    results = []
    with open(OUT_FILE, "w", encoding="utf-8") as fw:
        for rec in tqdm(data, desc="Reconstructing", unit="sample"):
            sample_id = rec.get("id", "")
            segments = extract_segments(rec)
            
            if not segments:
                out = {
                    "id": sample_id,
                    "pred": "",
                    "ori": rec.get("ori", ""),
                }
                for i, seg in enumerate(segments, 1):
                    out[f"seg{i}"] = seg
                fw.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue
            
            try:
                reconstructed, info = dsr.reconstruct(
                    segments=segments,
                    steps=STEPS,
                    gen_length=GEN_LENGTH,
                    block_length=BLOCK_LENGTH,
                    temperature=TEMPERATURE,
                    cfg_scale=CFG_SCALE,
                    remasking='lexical_aware' if (USE_LEXMASKER if USE_LEXMASKER is not None else True) else 'low_confidence'
                )
                
                out = {
                    "id": sample_id,
                    "pred": reconstructed.strip(),
                    "ori": rec.get("ori", ""),
                }
                
                for i, seg in enumerate(segments, 1):
                    out[f"seg{i}"] = seg
                
                if "lexmasker_applied" in info:
                    out["lexmasker_steps"] = info["lexmasker_applied"]
                
                fw.write(json.dumps(out, ensure_ascii=False) + "\n")
                results.append((sample_id, reconstructed, info))
                
            except Exception as e:
                out = {
                    "id": sample_id,
                    "pred": "",
                    "ori": rec.get("ori", ""),
                    "error": str(e)
                }
                for i, seg in enumerate(segments, 1):
                    out[f"seg{i}"] = seg
                fw.write(json.dumps(out, ensure_ascii=False) + "\n")
    
    print(f"Results written to: {OUT_FILE}")


if __name__ == "__main__":
    main()
