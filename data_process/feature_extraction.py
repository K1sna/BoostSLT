#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, '')
from pytorch_i3d import InceptionI3d

DATA_DIR = Path("")
VIDEO_DIR = None
OUTPUT_DIR = Path("")
I3D_WEIGHTS_PATH = Path("")

WINDOW_SIZE = None
STRIDE = None
FEATURE_DIM = None


def load_video_frames(video_path, start_frame=0, num_frames=-1):
    vidcap = cv2.VideoCapture(str(video_path))
    
    if not vidcap.isOpened():
        return None
    
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames == -1:
        num_frames = total_frames - start_frame
    
    frames = []
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for offset in range(min(num_frames, total_frames - start_frame)):
        success, img = vidcap.read()
        if not success:
            break
        
        w, h, c = img.shape
        if w < 224 or h < 224:
            scale = max(224.0 / w, 224.0 / h)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        if w != 256 or h != 256:
            img = cv2.resize(img, (256, 256))
        
        img = (img.astype(np.float32) / 255.0) * 2 - 1
        frames.append(img)
    
    vidcap.release()
    return np.asarray(frames, dtype=np.float32) if frames else None


def video_to_tensor(frames):
    return torch.from_numpy(frames.transpose([3, 0, 1, 2]))


def extract_features_with_sliding_window(model, video_frames, window_size=8, stride=2):
    if video_frames is None or len(video_frames) == 0:
        return None
    
    T = len(video_frames)
    
    if T < window_size:
        padding = window_size - T
        padded_frames = np.concatenate([video_frames, np.tile(video_frames[-1:], (padding, 1, 1, 1))], axis=0)
        video_frames = padded_frames
        T = len(video_frames)
    
    num_windows = max(1, (T - window_size) // stride + 1)
    
    video_tensor = video_to_tensor(video_frames)
    video_tensor = video_tensor.unsqueeze(0)
    
    all_features = []
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        if end_idx > T:
            start_idx = T - window_size
            end_idx = T
        
        window = video_tensor[:, :, start_idx:end_idx, :, :]
        
        with torch.no_grad():
            features = model.extract_features(window.cuda())
            features = features.mean(dim=2).squeeze(-1).squeeze(-1)
            all_features.append(features.cpu())
    
    if len(all_features) == 0:
        return None
    
    all_features = torch.cat(all_features, dim=0)
    
    return all_features


def find_video_file(sample_name, video_dir):
    if video_dir is None:
        return None
    
    video_dir = Path(video_dir)
    
    possible_names = [
        f"{sample_name}.mp4",
        f"{sample_name}.avi",
        f"{sample_name}.mov",
    ]
    
    for name in possible_names:
        video_path = video_dir / name
        if video_path.exists():
            return video_path
    
    for subdir in video_dir.iterdir():
        if subdir.is_dir():
            for name in possible_names:
                video_path = subdir / name
                if video_path.exists():
                    return video_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Extract I3D features')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Video directory path')
    parser.add_argument('--data_file', type=str, 
                       default="",
                       help='Data file path')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                       help='Feature output directory')
    parser.add_argument('--weights', type=str, default=str(I3D_WEIGHTS_PATH),
                       help='I3D model weights path')
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'],
                       default='train', help='Dataset split')
    parser.add_argument('--window_size', type=int, default=None,
                       help='Window size')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride')
    
    args = parser.parse_args()
    
    global VIDEO_DIR, OUTPUT_DIR, I3D_WEIGHTS_PATH, WINDOW_SIZE, STRIDE
    VIDEO_DIR = Path(args.video_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    I3D_WEIGHTS_PATH = Path(args.weights)
    WINDOW_SIZE = args.window_size if args.window_size else 8
    STRIDE = args.stride if args.stride else 2
    
    import gzip
    import pickle
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"Error: Data file does not exist: {data_file}")
        sys.exit(1)
    
    with gzip.open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    if not I3D_WEIGHTS_PATH.exists():
        print(f"Error: I3D weights file does not exist: {I3D_WEIGHTS_PATH}")
        sys.exit(1)
    
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(str(I3D_WEIGHTS_PATH)))
    i3d.cuda()
    i3d.eval()
    
    success_count = 0
    failed_count = 0
    missing_video_count = 0
    
    for sample in tqdm(data, desc="Extracting features"):
        sample_name = sample['name']
        
        video_path = find_video_file(sample_name, VIDEO_DIR)
        if video_path is None:
            missing_video_count += 1
            continue
        
        video_frames = load_video_frames(video_path)
        if video_frames is None:
            failed_count += 1
            continue
        
        try:
            features = extract_features_with_sliding_window(
                i3d, video_frames, window_size=WINDOW_SIZE, stride=STRIDE
            )
            
            if features is None:
                failed_count += 1
                continue
            
            output_file = OUTPUT_DIR / f"{sample_name}.pt"
            torch.save(features, output_file)
            success_count += 1
            
        except Exception as e:
            print(f"\nFeature extraction failed {sample_name}: {e}")
            failed_count += 1
            continue
    
    print(f"Feature extraction complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Missing video files: {missing_video_count}")
    print(f"  Features saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
