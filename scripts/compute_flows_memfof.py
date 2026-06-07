#!/usr/bin/env python3
"""
Compute optical flow for GoPro dataset using MEMFOF.
Generates flow_cp (backward: sharp(t) -> sharp(t-1)) and flow_cn (forward: sharp(t) -> sharp(t+1))
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image
import argparse

try:
    from memfof.model import MEMFOF
    MEMFOF_AVAILABLE = True
except ImportError:
    print("Warning: MEMFOF not installed. Install with: uv pip install -e memfof/")
    MEMFOF_AVAILABLE = False
    sys.exit(1)


def load_image(path):
    """Load image and normalize to [-1, 1]"""
    img = Image.open(path).convert('RGB')
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0) * 2.0 - 1.0  # [-1, 1]
    return img.transpose(2, 0, 1)  # [C, H, W]


def save_flow(flow, path):
    """Save flow as .npy file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, flow)


def flow_dirs_for_radius(video_dir, radius):
    if radius == 1:
        return os.path.join(video_dir, "flow_cp_pix"), os.path.join(video_dir, "flow_cn_pix")
    return os.path.join(video_dir, f"flow_cp{radius}_pix"), os.path.join(video_dir, f"flow_cn{radius}_pix")


def compute_flows_for_video(video_dir, model, device, split="train", limit_frames=None, skip_existing=True, radii=(1,)):
    """Compute flows for a single video directory"""
    sharp_dir = os.path.join(video_dir, "sharp")
    
    sharp_frames = sorted(glob(os.path.join(sharp_dir, "*.png")))
    
    if len(sharp_frames) < 3:
        print(f"Warning: {video_dir} has less than 3 frames, skipping")
        return
    
    print(f"Processing {len(sharp_frames)} frames in {video_dir}")
    
    for radius in radii:
        flow_cp_dir, flow_cn_dir = flow_dirs_for_radius(video_dir, radius)
        os.makedirs(flow_cp_dir, exist_ok=True)
        os.makedirs(flow_cn_dir, exist_ok=True)

        # Process frames in triplets: (prev, curr, next). For radius=2 this is
        # (t-2, t, t+2), giving the five-point trajectory endpoints.
        frame_indices = range(radius, len(sharp_frames) - radius)
        if limit_frames is not None:
            frame_indices = list(frame_indices)[:limit_frames]
        desc = f"{os.path.basename(video_dir)} r={radius}"
        for i in tqdm(frame_indices, desc=desc, leave=False):
            prev_path = sharp_frames[i - radius]
            curr_path = sharp_frames[i]
            next_path = sharp_frames[i + radius]

            frame_name = os.path.basename(curr_path).replace('.png', '.npy')
            flow_cp_path = os.path.join(flow_cp_dir, frame_name)
            flow_cn_path = os.path.join(flow_cn_dir, frame_name)
            if skip_existing and os.path.exists(flow_cp_path) and os.path.exists(flow_cn_path):
                continue
        
            # Load images (already normalized to [-1, 1])
            prev_img = load_image(prev_path)
            curr_img = load_image(curr_path)
            next_img = load_image(next_path)

            # Convert from [-1, 1] to [0, 255] for MEMFOF.
            prev_img = ((prev_img + 1.0) / 2.0 * 255.0).astype(np.uint8)
            curr_img = ((curr_img + 1.0) / 2.0 * 255.0).astype(np.uint8)
            next_img = ((next_img + 1.0) / 2.0 * 255.0).astype(np.uint8)

            frames = np.stack([prev_img, curr_img, next_img], axis=0)  # [3, C, H, W]
            frames_tensor = torch.from_numpy(frames).unsqueeze(0).to(device).float()  # [1, 3, C, H, W]

            with torch.inference_mode():
                result = model(frames_tensor)
                flows = result["flow"][-1]

            backward_flow = flows[:, 0]  # curr -> prev
            forward_flow = flows[:, 1]   # curr -> next

            flow_cp = backward_flow[0].cpu().numpy().transpose(1, 2, 0)
            flow_cn = forward_flow[0].cpu().numpy().transpose(1, 2, 0)

            save_flow(flow_cp, flow_cp_path)
            save_flow(flow_cn, flow_cn_path)


def main():
    parser = argparse.ArgumentParser(description="Compute optical flow for GoPro dataset")
    parser.add_argument("--data_path", type=str, default="./datasets/GoPro", help="Path to GoPro dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"], help="Dataset split")
    parser.add_argument("--model_name", type=str, default="egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH", 
                        help="MEMFOF model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    parser.add_argument("--limit-videos", type=int, default=None, help="Optional number of videos for smoke runs")
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional number of centre frames per video")
    parser.add_argument("--radii", type=str, default="1", help="Comma-separated temporal radii, e.g. 1,2")
    parser.add_argument("--no-skip-existing", action="store_true", help="Recompute flows even if .npy files exist")
    args = parser.parse_args()
    
    if not MEMFOF_AVAILABLE:
        print("Error: MEMFOF is not installed. Please install it first:")
        print("  pip install git+https://github.com/msu-video-group/memfof")
        sys.exit(1)
    
    # Map model names to checkpoint files
    model_to_ckpt = {
        "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH": "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH",
        "MEMFOF-Tartan": "Tartan.ckpt",
        "MEMFOF-Tartan-T": "Tartan-T.ckpt",
        "MEMFOF-Tartan-T-TSKH": "Tartan-T-TSKH.ckpt",
        "MEMFOF-Tartan-T-TSKH-kitti": "Tartan-T-TSKH-kitti.ckpt",
        "MEMFOF-Tartan-T-TSKH-sintel": "Tartan-T-TSKH-sintel.ckpt",
        "MEMFOF-Tartan-T-TSKH-spring": "Tartan-T-TSKH-spring.ckpt",
    }
    
    # Load MEMFOF model from local checkpoint <- раскомментируйте, если у вас своя модель
    #ckpt_name = model_to_ckpt.get(args.model_name)
    #if ckpt_name is None:
    #    print(f"Error: Unknown model name: {args.model_name}")
    #    print(f"Available models: {list(model_to_ckpt.keys())}")
    #    sys.exit(1)
    
    #ckpt_path = os.path.join("memfof", "ckpts", ckpt_name)
    #if not os.path.exists(ckpt_path):
    #    print(f"Error: Checkpoint not found: {ckpt_path}")
    #    sys.exit(1)
    
    #print(f"Loading MEMFOF model from local checkpoint: {ckpt_path}")
    

    # Initialize model (use default config, will load weights from checkpoint)
    model = MEMFOF.from_pretrained(args.model_name).eval().to(args.device)
    
    # Load checkpoint
    #checkpoint = torch.load(ckpt_path, map_location=args.device)
    
    # Handle different checkpoint formats (Lightning vs direct state_dict)
    '''if "state_dict" in checkpoint:
        # PyTorch Lightning format
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix if present (Lightning adds it)
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    elif "model" in checkpoint:
        # Nested model dict
        state_dict = checkpoint["model"]
    else:
        # Direct state_dict
        state_dict = checkpoint
    
    # Load state dict (use strict=False to handle any missing keys)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys (showing first 5): {missing_keys[:5]}")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys (showing first 5): {unexpected_keys[:5]}")
    print(f"Model loaded on {args.device}")'''
    
    radii = tuple(int(item) for item in args.radii.split(",") if item.strip())
    if not radii or any(radius < 1 for radius in radii):
        raise ValueError(f"Invalid --radii value: {args.radii}")

    # Process all videos
    split_path = os.path.join(args.data_path, args.split)
    if not os.path.exists(split_path):
        print(f"Error: Split path does not exist: {split_path}")
        sys.exit(1)
    
    video_dirs = sorted([os.path.join(split_path, d) for d in os.listdir(split_path) 
                        if os.path.isdir(os.path.join(split_path, d))])
    if args.limit_videos is not None:
        video_dirs = video_dirs[:args.limit_videos]
    
    print(f"Found {len(video_dirs)} videos in {split_path}")
    
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        try:
            compute_flows_for_video(
                video_dir,
                model,
                args.device,
                args.split,
                limit_frames=args.limit_frames,
                skip_existing=not args.no_skip_existing,
                radii=radii,
            )
        except Exception as e:
            print(f"Error processing {video_dir}: {e}")
            continue
    
    print("Flow computation complete!")


if __name__ == "__main__":
    main()
