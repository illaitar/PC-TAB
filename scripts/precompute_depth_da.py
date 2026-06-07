"""Precompute Depth Anything 3 depth, intrinsics, and camera poses.

The paper-side PC-TAB method expects per-triplet depth and camera parameters.
This script writes:

- depth/{center}.npy: centre-frame depth
- camera/{center}.npz: DA3 triplet extrinsics [3,3,4], intrinsics [3,3,3],
  confidence if available, and source frame paths

If Depth Anything 3 is unavailable, use `--allow-v2-fallback` to write
monocular Depth-Anything-V2 depths only. The fallback is useful for development
but does not fully match the paper method because it has no pose/intrinsics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def device_default() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def triplets(data_root: Path, split: str, limit: int) -> list[tuple[Path, Path, Path]]:
    out: list[tuple[Path, Path, Path]] = []
    for video in sorted((data_root / split).iterdir()):
        if not video.is_dir():
            continue
        frames = sorted((video / "sharp").glob("*.png"))
        for idx in range(1, len(frames) - 1):
            out.append((frames[idx - 1], frames[idx], frames[idx + 1]))
            if limit > 0 and len(out) >= limit:
                return out
    return out


def load_da3(model_name: str, device: str):
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "Depth Anything 3 is not installed. Install ByteDance-Seed/Depth-Anything-3 "
            "or rerun with --allow-v2-fallback for depth-only development output."
        ) from exc
    model = DepthAnything3(model_name=model_name).to(device)
    return model


def run_da3_triplet(model, frames: list[Path], process_res: int):
    return model.inference(
        image=[str(path) for path in frames],
        ref_view_strategy="middle",
        process_res=process_res,
        export_format="mini_npz",
    )


def load_v2(model_size: str, device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return model, processor


@torch.no_grad()
def run_v2_depth(image_path: Path, model, processor, device: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    depth = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()
    depth = depth.astype(np.float32)
    depth -= depth.min()
    if depth.max() > 0:
        depth /= depth.max()
    return depth


def save_da3_outputs(prediction, frames: tuple[Path, Path, Path], overwrite: bool) -> dict:
    prev, center, next_frame = frames
    video = center.parent.parent
    stem = center.stem
    depth_dir = video / "depth"
    camera_dir = video / "camera"
    depth_dir.mkdir(exist_ok=True)
    camera_dir.mkdir(exist_ok=True)
    depth_path = depth_dir / f"{stem}.npy"
    camera_path = camera_dir / f"{stem}.npz"
    if depth_path.exists() and camera_path.exists() and not overwrite:
        return {"depth": str(depth_path), "camera": str(camera_path), "skipped": True}

    depth = np.asarray(prediction.depth, dtype=np.float32)
    center_idx = 1
    np.save(depth_path, depth[center_idx])
    payload = {
        "extrinsics": np.asarray(prediction.extrinsics, dtype=np.float32),
        "intrinsics": np.asarray(prediction.intrinsics, dtype=np.float32),
        "frames": np.array([str(prev), str(center), str(next_frame)]),
    }
    if hasattr(prediction, "conf"):
        payload["confidence"] = np.asarray(prediction.conf, dtype=np.float32)
    np.savez_compressed(camera_path, **payload)
    return {"depth": str(depth_path), "camera": str(camera_path), "skipped": False}


def save_v2_output(depth: np.ndarray, center: Path, overwrite: bool) -> dict:
    depth_dir = center.parent.parent / "depth"
    depth_dir.mkdir(exist_ok=True)
    depth_path = depth_dir / f"{center.stem}.npy"
    if depth_path.exists() and not overwrite:
        return {"depth": str(depth_path), "camera": None, "skipped": True}
    np.save(depth_path, depth.astype(np.float32))
    return {"depth": str(depth_path), "camera": None, "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/GoPro"))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=device_default())
    parser.add_argument("--model-name", default="da3-large")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-v2-fallback", action="store_true")
    parser.add_argument("--v2-model-size", default="large", choices=["small", "base", "large"])
    args = parser.parse_args()

    frames = triplets(args.data_root, args.split, args.limit)
    if not frames:
        raise RuntimeError(f"No triplets found under {args.data_root / args.split}")

    manifest = {
        "split": args.split,
        "data_root": str(args.data_root),
        "estimator": "DepthAnything3",
        "model_name": args.model_name,
        "process_res": args.process_res,
        "items": [],
    }
    try:
        da3 = load_da3(args.model_name, args.device)
        for item in tqdm(frames, desc=f"DA3 {args.split}"):
            pred = run_da3_triplet(da3, list(item), args.process_res)
            manifest["items"].append(save_da3_outputs(pred, item, args.overwrite))
    except ImportError:
        if not args.allow_v2_fallback:
            raise
        manifest["estimator"] = "DepthAnythingV2-depth-only-fallback"
        manifest["model_name"] = args.v2_model_size
        model, processor = load_v2(args.v2_model_size, args.device)
        for _, center, _ in tqdm(frames, desc=f"DA-V2 fallback {args.split}"):
            depth = run_v2_depth(center, model, processor, args.device)
            manifest["items"].append(save_v2_output(depth, center, args.overwrite))

    out_path = args.data_root / args.split / "depth_camera_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {out_path}")


if __name__ == "__main__":
    main()
