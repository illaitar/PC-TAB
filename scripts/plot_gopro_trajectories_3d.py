"""Render 3D trajectory plots for the GoPro PC-TAB subset collage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from pc_tab_runtime import load_pc_tab_impl
from run_gopro_pctab_subset import (
    depth_proxy,
    find_triplets,
    image_to_tensor,
    make_params,
    presets,
    write_tensor_image,
)


def plot_trajectories_3d(
    traj: torch.Tensor,
    output: Path,
    title: str,
    grid_rows: int = 7,
    grid_cols: int = 10,
) -> None:
    """Plot sampled pixel trajectories as (x, y, exposure time) curves."""

    traj_np = traj.detach().cpu().float().numpy()
    if traj_np.ndim == 5:
        traj_np = traj_np[0]
    num_steps, height, width, _ = traj_np.shape

    ys = np.linspace(height * 0.18, height * 0.82, grid_rows).astype(int)
    xs = np.linspace(width * 0.12, width * 0.88, grid_cols).astype(int)
    times = np.linspace(-1.0, 1.0, num_steps)
    projection_z = -1.18

    fig = plt.figure(figsize=(7.0, 5.2), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")

    max_disp = float(np.linalg.norm(traj_np, axis=-1).max())
    for row_idx, y in enumerate(ys):
        for x in xs:
            xy = traj_np[:, y, x, :]
            px = x + xy[:, 0]
            py = y + xy[:, 1]
            color = cmap(row_idx / max(len(ys) - 1, 1))
            ax.plot(px, py, times, color=color, linewidth=0.9, alpha=0.78)
            ax.plot(
                px,
                py,
                np.full_like(times, projection_z),
                color=color,
                linewidth=0.7,
                alpha=0.28,
                linestyle="--",
            )
            ax.scatter(px[0], py[0], times[0], color=color, s=4, alpha=0.55)

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("x + d_x(t)", labelpad=5)
    ax.set_ylabel("y + d_y(t)", labelpad=5)
    ax.set_zlabel("t", labelpad=5)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_zlim(projection_z, 1)
    ax.view_init(elev=33, azim=-66)
    ax.text2D(0.02, 0.95, f"max |d|={max_disp:.1f}px", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_pair_panel(label: str, blur_path: Path, trajectory_path: Path, output: Path) -> None:
    blur = Image.open(blur_path).convert("RGB")
    traj = Image.open(trajectory_path).convert("RGB")
    blur.thumbnail((360, 240), Image.Resampling.LANCZOS)
    traj.thumbnail((360, 270), Image.Resampling.LANCZOS)

    width = 760
    height = 330
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), label, fill=(0, 0, 0))
    canvas.paste(blur, (14, 48))
    canvas.paste(traj, (390, 38))
    draw.text((14, 292), "rendered blur", fill=(60, 60, 60))
    draw.text((390, 292), "3D exposure trajectories", fill=(60, 60, 60))
    canvas.save(output)


def make_collage(panels: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in panels]
    cols = 1
    rows = len(images)
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (cols * width, height), "white")
    y = 0
    for image in images:
        sheet.paste(image, ((width - image.width) // 2, y))
        y += image.height
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/GoPro"))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--out", type=Path, default=Path("outputs/gopro_trajectory_3d"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--grid-rows", type=int, default=7)
    parser.add_argument("--grid-cols", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    pc_tab = load_pc_tab_impl()
    cfg = pc_tab.SIN3DConfig(
        device=str(device),
        num_subframes=24,
        use_forward_splat_visibility=False,
        camera_model="homography",
    )
    engine = pc_tab.SIN3DEngine(cfg, extract_objects=False, use_linear_light=True, use_depth_ordering=True)
    base_params = cfg.sample()
    triplets = find_triplets(args.data_root, args.split, args.limit)
    if not triplets:
        raise RuntimeError(f"No triplets found under {args.data_root / args.split}")

    manifest = {
        "data_root": str(args.data_root),
        "split": args.split,
        "max_side": args.max_side,
        "samples": [],
    }
    panel_paths: list[Path] = []

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

        center_path_out = sample_dir / "center_sharp.png"
        write_tensor_image(sharp_seq[1], center_path_out)
        sample_info = {
            "index": sample_idx,
            "prev": str(prev_path),
            "center": str(center_path),
            "next": str(next_path),
            "variants": [],
        }

        for variant_name, overrides in presets():
            params = make_params(base_params, overrides)
            params.device = str(device)
            blur, meta = engine.synthesize(
                sharp_seq=sharp_seq,
                depth=depth,
                flow_fwd=zero_flow,
                flow_bwd=zero_flow,
                params=params,
            )

            blur_path = sample_dir / f"{variant_name}_blur.png"
            traj_path = sample_dir / f"{variant_name}_trajectory_3d.png"
            panel_path = sample_dir / f"{variant_name}_panel.png"
            write_tensor_image(blur, blur_path)
            plot_trajectories_3d(
                meta["traj"],
                traj_path,
                f"sample {sample_idx}: {variant_name}",
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
            )
            make_pair_panel(f"sample {sample_idx} / {variant_name}", blur_path, traj_path, panel_path)
            panel_paths.append(panel_path)

            sample_info["variants"].append(
                {
                    "name": variant_name,
                    "blur": str(blur_path),
                    "trajectory_3d": str(traj_path),
                    "panel": str(panel_path),
                    "params": {key: value for key, value in asdict(params).items() if not key.startswith("_")},
                    "trajectory_abs_mean": float(meta["traj"].abs().mean().detach().cpu()),
                    "trajectory_abs_max": float(meta["traj"].abs().max().detach().cpu()),
                    "visibility_mean": float(meta["visibility"].mean().detach().cpu()),
                }
            )
        manifest["samples"].append(sample_info)

    make_collage(panel_paths, args.out / "trajectory_3d_collage.png")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote 3D trajectory plots to {args.out}")
    print(f"Collage: {args.out / 'trajectory_3d_collage.png'}")


if __name__ == "__main__":
    main()
