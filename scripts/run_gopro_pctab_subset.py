"""Generate PC-TAB blur samples from a local GoPro subset.

This runner uses real GoPro sharp triplets and the paper-aligned implementation
in `PC-TABD-main/pc-tab`. It is intentionally lightweight: camera motion is
estimated by the implementation's homography fallback, residual flow is zero,
and depth is a smooth proxy. Use `compute_flows_memfof.py` and
`precompute_depth_da.py` before this script when MEMFOF/DA depth is available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

from pc_tab_runtime import load_pc_tab_impl


def image_to_tensor(path: Path, max_side: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if max_side > 0:
        scale = max(image.size) / float(max_side)
        if scale > 1.0:
            image = image.resize(
                (round(image.width / scale), round(image.height / scale)),
                Image.Resampling.LANCZOS,
            )
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def write_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.detach().clamp(0.0, 1.0).cpu()
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def find_triplets(data_root: Path, split: str, limit: int) -> list[tuple[Path, Path, Path]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Missing GoPro split folder: {split_root}")

    triplets: list[tuple[Path, Path, Path]] = []
    for video in sorted(p for p in split_root.iterdir() if p.is_dir()):
        sharp_dir = video / "sharp"
        if not sharp_dir.exists():
            continue
        frames = sorted(sharp_dir.glob("*.png"))
        for idx in range(1, len(frames) - 1):
            triplets.append((frames[idx - 1], frames[idx], frames[idx + 1]))
            if len(triplets) >= limit:
                return triplets
    return triplets


def depth_proxy(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height, device=device).view(1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, width)
    depth = 0.8 + 0.35 * y + 0.05 * torch.sin(2.0 * torch.pi * x)
    return depth.unsqueeze(0).clamp(min=0.1)


def presets() -> Iterable[tuple[str, dict]]:
    yield (
        "clean_triangle",
        {
            "shutter_length": 1.0,
            "shutter_profile": "triangle",
            "rolling_shutter_strength": 0.0,
            "camera_translation_scale": 1.0,
            "object_scale": 0.0,
            "noise_level": 0.0,
            "noise_poisson_scale": 0.0,
            "motion_sharpening": 0.0,
            "depth_parallax_scale": 1.0,
            "trajectory_profile": "constant",
            "camera_acceleration": 0.0,
            "lateral_acceleration": 0.0,
        },
    )
    yield (
        "strong_box_rs",
        {
            "shutter_length": 1.35,
            "shutter_profile": "box",
            "rolling_shutter_strength": 0.18,
            "camera_translation_scale": 1.35,
            "object_scale": 0.0,
            "noise_level": 0.0,
            "noise_poisson_scale": 0.0,
            "motion_sharpening": 0.0,
            "depth_parallax_scale": 1.0,
            "trajectory_profile": "constant",
            "camera_acceleration": 0.0,
            "lateral_acceleration": 0.0,
        },
    )
    yield (
        "accelerated_gaussian",
        {
            "shutter_length": 1.2,
            "shutter_profile": "gaussian",
            "rolling_shutter_strength": 0.08,
            "camera_translation_scale": 1.1,
            "object_scale": 0.0,
            "noise_level": 0.001,
            "noise_poisson_scale": 0.001,
            "motion_sharpening": 0.02,
            "depth_parallax_scale": 1.0,
            "trajectory_profile": "acceleration",
            "camera_acceleration": 0.18,
            "lateral_acceleration": 1.8,
        },
    )


def make_params(base_params, overrides: dict):
    fields = set(base_params.__dataclass_fields__.keys())
    return replace(base_params, **{key: value for key, value in overrides.items() if key in fields})


def make_contact_sheet(items: list[tuple[str, Path]], output: Path) -> None:
    thumbs = []
    for label, path in items:
        image = Image.open(path).convert("RGB")
        image.thumbnail((300, 190), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (320, 232), "white")
        canvas.paste(image, ((320 - image.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 208), label[:42], fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 320, rows * 232), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 320, (idx // cols) * 232))
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/GoPro"))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--out", type=Path, default=Path("outputs/gopro_pctab_subset"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    triplets = find_triplets(args.data_root, args.split, args.limit)
    if not triplets:
        raise RuntimeError(f"No GoPro triplets found under {args.data_root / args.split}")

    pc_tab = load_pc_tab_impl()
    cfg = pc_tab.SIN3DConfig(
        device=str(device),
        num_subframes=24,
        use_forward_splat_visibility=False,
        camera_model="homography",
    )
    engine = pc_tab.SIN3DEngine(cfg, extract_objects=False, use_linear_light=True, use_depth_ordering=True)
    base_params = cfg.sample()

    manifest = {
        "data_root": str(args.data_root),
        "split": args.split,
        "device": str(device),
        "max_side": args.max_side,
        "note": "GoPro smoke run with homography camera extraction, proxy depth, zero MEMFOF residual flow.",
        "samples": [],
    }
    sheet_items: list[tuple[str, Path]] = []

    for sample_idx, (prev_path, center_path, next_path) in enumerate(triplets):
        sample_dir = args.out / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sharp_seq = torch.stack(
            [
                image_to_tensor(prev_path, args.max_side, device),
                image_to_tensor(center_path, args.max_side, device),
                image_to_tensor(next_path, args.max_side, device),
            ],
            dim=0,
        )
        _, height, width = sharp_seq[1].shape
        depth = depth_proxy(height, width, device)
        zero_flow = torch.zeros(2, height, width, device=device)

        center_out = sample_dir / "center_sharp.png"
        write_tensor_image(sharp_seq[1], center_out)
        sheet_items.append((f"s{sample_idx} sharp", center_out))

        sample_manifest = {
            "index": sample_idx,
            "prev": str(prev_path),
            "center": str(center_path),
            "next": str(next_path),
            "variants": [],
        }
        for name, overrides in presets():
            params = make_params(base_params, overrides)
            params.device = str(device)
            blur, meta = engine.synthesize(
                sharp_seq=sharp_seq,
                depth=depth,
                flow_fwd=zero_flow,
                flow_bwd=zero_flow,
                params=params,
            )
            out_path = sample_dir / f"{name}.png"
            write_tensor_image(blur, out_path)
            sheet_items.append((f"s{sample_idx} {name}", out_path))
            sample_manifest["variants"].append(
                {
                    "name": name,
                    "output": str(out_path),
                    "params": {k: v for k, v in asdict(params).items() if not k.startswith("_")},
                    "trajectory_abs_mean": float(meta["traj"].abs().mean().detach().cpu()),
                    "trajectory_abs_max": float(meta["traj"].abs().max().detach().cpu()),
                    "visibility_mean": float(meta["visibility"].mean().detach().cpu()),
                }
            )
        manifest["samples"].append(sample_manifest)

    make_contact_sheet(sheet_items, args.out / "contact_sheet.png")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(triplets)} samples to {args.out}")
    print(f"Contact sheet: {args.out / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
