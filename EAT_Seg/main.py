#!/usr/bin/env python3

import pickle
import gzip
from pathlib import Path
import numpy as np
import torch 
import sys
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit='it', **kwargs):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if iterable else 100)
            self.desc = desc or ""
            self.unit = unit
            self.current = 0
            self.start_time = time.time()
            self.last_update = 0
            
        def __iter__(self):
            return self
            
        def __next__(self):
            if self.current >= self.total:
                raise StopIteration
            item = self.iterable[self.current] if self.iterable else self.current
            self.current += 1
            self.update(1)
            return item
        
        def update(self, n=1):
            self.current = min(self.current + n, self.total)
            now = time.time()
            if now - self.last_update >= 0.5 or self.current >= self.total:
                self.last_update = now
                self._display()
        
        def _display(self):
            percent = 100 * self.current / self.total if self.total > 0 else 0
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            bar_length = 40
            filled = int(bar_length * self.current / self.total) if self.total > 0 else 0
            bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
            
            info = f"\r{self.desc} |{bar}| {self.current}/{self.total} [{percent:.1f}%] "
            info += f"{rate:.1f}{self.unit}/s, ETA: {eta:.1f}s"
            sys.stdout.write(info)
            sys.stdout.flush()
            
            if self.current >= self.total:
                sys.stdout.write('\n')
                sys.stdout.flush()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.current < self.total:
                self.current = self.total
                self._display()
            
        def close(self):
            if self.current < self.total:
                self.current = self.total
                self._display()

PATHS = {
    "pkl": "",
    "data_dev": "",
    "data_test": "",
    "data_train": "",
    "out_dev": "",
    "out_test": "",
    "out_train": "",
    "debug_dir": None,
}

HYPERS = {
    "mode": None,
    "use_hand_only": None,
    "conf_th": None,
    "smooth_k": None,
    "Lmin": None,
    "Lstar": None,
    "Lmax": None,
    "window_factor": None,
    "lambda_center": None,
    "min_gap_ratio": None,
    "min_last_ratio": None,
    "overlap": None,
    "thr_fixed": None,
    "thr_quantile": None,
    "min_len_threshold": None,
    "merge_gap": None,
}

WB = {"lhand21": None, "rhand21": None}


def parse_key(key: str):
    parts = key.split("/")
    if len(parts) == 1:
        return "unknown", parts[0]
    return parts[0], parts[-1]


def make_segment_id(name: str, i: int, m: int):
    if "/" in name:
        stem = name.split("/")[-1]
    else:
        stem = name
    return f"{stem}_n{i}_m{m}"


def load_dataset_file(filepath: str):
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def segment_feature(sign, start_frame: int, end_frame: int, total_frames: int = None):
    if sign is None:
        return None
    
    sign_len = None
    if isinstance(sign, torch.Tensor):
        if sign.shape[0] == 0:
            return sign
        sign_len = sign.shape[0]
    elif isinstance(sign, np.ndarray):
        if sign.shape[0] == 0:
            return sign
        sign_len = sign.shape[0]
    elif hasattr(sign, '__len__'):
        sign_len = len(sign)
        if sign_len == 0:
            return sign
    
    if sign_len is None:
        return sign
    
    if total_frames is not None and sign_len != total_frames and sign_len > 0:
        ratio_start = start_frame / total_frames if total_frames > 0 else 0
        ratio_end = end_frame / total_frames if total_frames > 0 else 1
        idx_start = int(ratio_start * sign_len)
        idx_end = int(ratio_end * sign_len)
        idx_start = max(0, min(idx_start, sign_len - 1))
        idx_end = max(idx_start + 1, min(idx_end, sign_len))
    else:
        idx_start = start_frame
        idx_end = end_frame
        idx_start = max(0, min(idx_start, sign_len - 1))
        idx_end = max(idx_start + 1, min(idx_end, sign_len))
    
    if isinstance(sign, torch.Tensor):
        return sign[idx_start:idx_end].clone()
    elif isinstance(sign, np.ndarray):
        return sign[idx_start:idx_end].copy()
    elif hasattr(sign, '__getitem__'):
        return sign[idx_start:idx_end]
    else:
        return sign


def segment_gloss(gloss, start_frame: int, end_frame: int, total_frames: int):
    if gloss is None:
        return None
    
    if isinstance(gloss, list):
        if len(gloss) == total_frames:
            return gloss[start_frame:end_frame]
        elif len(gloss) > 0:
            ratio_start = start_frame / total_frames if total_frames > 0 else 0
            ratio_end = end_frame / total_frames if total_frames > 0 else 1
            idx_start = int(ratio_start * len(gloss))
            idx_end = int(ratio_end * len(gloss))
            return gloss[idx_start:idx_end]
    
    return gloss


def smooth(x, k=9):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def energy(seq, conf_th=0.2, slice_=None):
    xy = seq[..., :2].astype(np.float32)
    cf = seq[..., 2].astype(np.float32)
    if slice_ is not None:
        xy = xy[:, slice_, :]
        cf = cf[:, slice_]
    T = xy.shape[0]
    E = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        m = (cf[t] >= conf_th) & (cf[t - 1] >= conf_th)
        if np.any(m):
            d = xy[t, m] - xy[t - 1, m]
            w = cf[t, m] * cf[t - 1, m]
            E[t] = np.sum(np.linalg.norm(d, axis=1) * w)
    return E


def find_segments_threshold(Es, thr, min_len=15, merge_gap=6):
    T = len(Es)
    active = (Es >= thr).astype(np.int32)
    segments, s = [], None
    for t in range(T):
        if active[t] and s is None:
            s = t
        if not active[t] and s is not None:
            e = t
            if e - s >= min_len:
                segments.append([s, e])
            s = None
    if s is not None:
        e = T
        if e - s >= min_len:
            segments.append([s, e])

    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        if seg[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return merged


def decide_n(F, Lmin, Lstar, Lmax):
    if F <= Lmax:
        return 1
    n = max(1, int(round(F / max(Lstar, 1))))
    while n > 1 and F / n > Lmax:
        n += 1
    while F / n < Lmin and n > 1:
        n -= 1
    n = max(1, n)
    return n


def distribute_targets(F, n, Lmin, Lmax):
    avg = F / n
    L_tgt = np.clip(avg, Lmin, Lmax)
    lengths = np.full(n, int(round(L_tgt)), dtype=int)
    diff = int(F - lengths.sum())
    i = 0
    step = 1 if diff > 0 else -1
    while diff != 0 and n > 0:
        lengths[i % n] += step
        diff -= step
        i += 1
    return lengths.tolist()


def pick_cut_in_window(E, center, left, right, lam=0.02):
    L = max(left, 0)
    R = min(right, len(E) - 1)
    xs = np.arange(L, R + 1)
    cost = E[xs] + lam * np.abs(xs - center)
    idx = int(xs[np.argmin(cost)])
    return idx


def segment_length_aware(E, F, Lmin, Lstar, Lmax,
                         window_factor=0.15, lam_center=0.02,
                         min_gap_ratio=0.60, min_last_ratio=0.80):
    n = decide_n(F, Lmin, Lstar, Lmax)
    if n == 1:
        return [[0, F]], dict(n=n)

    lengths = distribute_targets(F, n, Lmin, Lmax)
    centers = np.cumsum(lengths[:-1])
    centers = centers - np.array(lengths[:-1]) // 2

    min_gap = int(round(min_gap_ratio * Lmin))
    min_tail = int(round(min_last_ratio * Lmin))

    cuts = []
    last_cut = 0
    for k, c in enumerate(centers, start=1):
        Lk = lengths[k - 1]
        win = max(3, int(round(window_factor * Lk)))
        left = max(last_cut + min_gap, c - win)
        right_limit = F - min_tail if k == (len(centers)) else F - 1
        right = min(c + win, right_limit)

        if left >= right:
            cand = max(last_cut + min_gap, min(c, right_limit))
        else:
            cand = pick_cut_in_window(E, c, left, right, lam=0.02 if lam_center is None else lam_center)

        cuts.append(int(cand))
        last_cut = cand

    segments = []
    s = 0
    for cut in cuts:
        e = max(cut, s)
        segments.append([s, e])
        s = e
    segments.append([s, F])

    if segments[-1][1] - segments[-1][0] < min_tail and len(segments) >= 2:
        k = len(segments) - 2
        prev_s, prev_e = segments[k][0], segments[k][1]
        target = prev_e - (min_tail)
        target = max(prev_s + min_gap, target)
        cand = pick_cut_in_window(E, target, prev_s + min_gap, prev_e - 1, lam=0.02 if lam_center is None else lam_center)
        segments[k][1] = cand
        segments[-1][0] = cand

    return segments, dict(n=n, lengths=lengths, centers=centers.tolist(), cuts=cuts)


def add_overlap(segments, overlap, T):
    if overlap <= 0:
        return segments
    segs = []
    for i, (s, e) in enumerate(segments):
        s2 = max(0, s - (overlap if i > 0 else 0))
        e2 = min(T, e + (overlap if i < len(segments) - 1 else 0))
        segs.append([s2, e2])
    return segs


def process_dataset_split(pkl_db, data_filepath, out_filepath, split_name):
    use_hand = HYPERS["use_hand_only"]
    conf_th = HYPERS["conf_th"]
    smooth_k = HYPERS["smooth_k"]

    mode = HYPERS["mode"]
    Lmin, Lstar, Lmax = HYPERS["Lmin"], HYPERS["Lstar"], HYPERS["Lmax"]
    window_factor = HYPERS["window_factor"]
    lambda_center = HYPERS["lambda_center"]
    min_gap_ratio = HYPERS["min_gap_ratio"]
    min_last_ratio = HYPERS["min_last_ratio"]
    overlap = HYPERS["overlap"]

    thr_fixed = HYPERS["thr_fixed"]
    thr_q = HYPERS["thr_quantile"]
    min_len_thr = HYPERS["min_len_threshold"]
    merge_gap = HYPERS["merge_gap"]

    print(f"[{split_name}] Loading dataset file: {data_filepath}")
    dataset = load_dataset_file(data_filepath)
    print(f"[{split_name}] Loaded {len(dataset)} samples")

    segmented_dataset = []
    n_total = 0
    n_missed_kp = 0
    n_segmented = 0

    pbar = tqdm(
        total=len(dataset),
        desc=f"[{split_name}] Processing",
        unit="sample"
    )

    for sample in dataset:
        name = sample.get("name", "")
        if not name:
            pbar.update(1)
            continue
        
        n_total += 1
        
        key = None
        for candidate_key in [name, name.replace("__", "/"), f"{split_name}/{name.split('/')[-1]}"]:
            if candidate_key in pkl_db:
                key = candidate_key
                break
        
        if key is None:
            n_missed_kp += 1
            segmented_dataset.append(sample)
            pbar.update(1)
            pbar.set_postfix({
                'Total': n_total,
                'Segmented': n_segmented,
                'Missing KP': n_missed_kp,
                'Generated': len(segmented_dataset)
            })
            continue

        item = pkl_db[key]
        kp = item["keypoints"]
        T = kp.shape[0]
        
        num_frames = sample.get("num_frames", T)
        if num_frames != T:
            T = min(num_frames, T)

        if use_hand:
            E = energy(kp, conf_th=conf_th, slice_=WB["lhand21"]) \
              + energy(kp, conf_th=conf_th, slice_=WB["rhand21"])
        else:
            E = energy(kp, conf_th=conf_th, slice_=None)

        Es = smooth(E, k=smooth_k)

        if mode == "threshold":
            thr = float(thr_fixed) if thr_fixed is not None else float(np.percentile(Es, thr_q))
            segs = find_segments_threshold(Es, thr=thr, min_len=min_len_thr, merge_gap=merge_gap)
        else:
            segs, _ = segment_length_aware(
                Es, T, Lmin=Lmin, Lstar=Lstar, Lmax=Lmax,
                window_factor=window_factor, lam_center=lambda_center,
                min_gap_ratio=min_gap_ratio, min_last_ratio=min_last_ratio
            )

        if len(segs) <= 1:
            segmented_dataset.append(sample)
            pbar.update(1)
            pbar.set_postfix({
                'Total': n_total,
                'Segmented': n_segmented,
                'Missing KP': n_missed_kp,
                'Generated': len(segmented_dataset)
            })
            continue

        m = len(segs)
        original_sign = sample.get("sign")
        original_gloss = sample.get("gloss")
        original_text = sample.get("text", "")
        original_signer = sample.get("signer", "")
        original_alignments = sample.get("alignments", None)

        for idx, (s, e) in enumerate(segs, start=1):
            new_name = make_segment_id(name, idx, m)
            
            seg_sign = segment_feature(original_sign, int(s), int(e), T)
            
            seg_gloss = segment_gloss(original_gloss, int(s), int(e), T)
            
            seg_alignments = None
            if original_alignments is not None and isinstance(original_alignments, list):
                if len(original_alignments) == T:
                    seg_alignments = original_alignments[int(s):int(e)]
                elif len(original_alignments) > 0:
                    ratio_start = int(s) / T if T > 0 else 0
                    ratio_end = int(e) / T if T > 0 else 1
                    idx_start = int(ratio_start * len(original_alignments))
                    idx_end = int(ratio_end * len(original_alignments))
                    seg_alignments = original_alignments[idx_start:idx_end]

            seg_sample = {
                "name": new_name,
                "signer": original_signer,
                "gloss": seg_gloss,
                "text": original_text,
                "sign": seg_sign,
                "num_frames": int(e) - int(s),
                "alignments": seg_alignments
            }
            
            segmented_dataset.append(seg_sample)
        
        n_segmented += 1
        
        pbar.update(1)
        pbar.set_postfix({
            'Total': n_total,
            'Segmented': n_segmented,
            'Missing KP': n_missed_kp,
            'Generated': len(segmented_dataset)
        })

    pbar.close()

    print(f"[{split_name}] Processing complete: Total={n_total}, Segmented={n_segmented}, Missing keypoints={n_missed_kp}")
    print(f"[{split_name}] Generated {len(segmented_dataset)} samples (original {len(dataset)})")
    
    out_path = Path(out_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with gzip.open(out_filepath, 'wb') as f:
        pickle.dump(segmented_dataset, f)
    
    print(f"[{split_name}] Saved to: {out_filepath}")


def main():
    pkl_path = Path(PATHS["pkl"])
    
    print("=" * 60)
    print(f"Loading keypoints database: {pkl_path}")
    with open(pkl_path, "rb") as f:
        pkl_db = pickle.load(f)
    print(f"Loaded {len(pkl_db)} keypoints samples")
    print("=" * 60)

    data_files = {
        "dev": (PATHS["data_dev"], PATHS["out_dev"]),
        "test": (PATHS["data_test"], PATHS["out_test"]),
        "train": (PATHS["data_train"], PATHS["out_train"])
    }

    print(f"\nStarting to process {len(data_files)} datasets...")
    print("-" * 60)
    
    split_pbar = tqdm(
        total=len(data_files),
        desc="Dataset progress",
        unit="dataset"
    )
    
    for split_name, (data_file, out_file) in data_files.items():
        process_dataset_split(pkl_db, data_file, out_file, split_name)
        split_pbar.update(1)
        split_pbar.set_postfix({'Current': split_name})
    
    split_pbar.close()
    print("\n" + "=" * 60)
    print("[DONE] All datasets processed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
