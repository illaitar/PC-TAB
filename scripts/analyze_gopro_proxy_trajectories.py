"""Estimate proxy GoPro trajectories and classify them into low-order classes.

This is a diagnostic approximation, not sensor ground truth. GoPro gives sharp
frame sequences, so we estimate dense optical flow from the centre frame to
neighbouring sharp frames and treat those correspondences as proxy point
positions over time. With five frames, we can separate:

- constant_velocity: well fit by a line p(t) = p0 + v t
- straight_acceleration: not constant velocity, but still geometrically straight
- curved_acceleration: well fit by p(t) = p0 + v t + 0.5 a t^2 with visible curvature
- other: not well explained by either low-order model
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class TrajectoryRecord:
    label: str
    positions: np.ndarray  # [T, 2]
    times: np.ndarray  # [T]
    linear_rms: float
    accel_rms: float
    span: float
    centre_xy: tuple[float, float]
    frame_path: str


def read_rgb(path: Path, max_side: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if max_side > 0:
        scale = max(image.size) / float(max_side)
        if scale > 1.0:
            image = image.resize(
                (round(image.width / scale), round(image.height / scale)),
                Image.Resampling.LANCZOS,
            )
    return np.asarray(image)


def to_gray_float(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def dense_flow(center_rgb: np.ndarray, target_rgb: np.ndarray, method: str) -> np.ndarray:
    """Flow from centre frame to target frame, [H,W,2] in pixels."""

    center_gray_u8 = cv2.cvtColor(center_rgb, cv2.COLOR_RGB2GRAY)
    target_gray_u8 = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    if method == "dis" and hasattr(cv2, "DISOpticalFlow_create"):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(center_gray_u8, target_gray_u8, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            center_gray_u8,
            target_gray_u8,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=25,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
    return flow.astype(np.float32)


def memfof_flow_paths_for_center(center_path: Path) -> list[Path]:
    video_dir = center_path.parent.parent
    frame_name = center_path.with_suffix(".npy").name
    return [
        video_dir / "flow_cp2_pix" / frame_name,
        video_dir / "flow_cp_pix" / frame_name,
        video_dir / "flow_cn_pix" / frame_name,
        video_dir / "flow_cn2_pix" / frame_name,
    ]


def find_windows(
    data_root: Path,
    split: str,
    limit: int,
    radius: int = 2,
    require_memfof: bool = False,
) -> list[list[Path]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(split_root)

    windows: list[list[Path]] = []
    for video in sorted(p for p in split_root.iterdir() if p.is_dir()):
        frames = sorted((video / "sharp").glob("*.png"))
        for idx in range(radius, len(frames) - radius):
            if require_memfof and not all(path.exists() for path in memfof_flow_paths_for_center(frames[idx])):
                continue
            windows.append(frames[idx - radius : idx + radius + 1])
            if limit > 0 and len(windows) >= limit:
                return windows
    return windows


def fit_rms(times: np.ndarray, positions: np.ndarray, degree: int) -> tuple[float, np.ndarray]:
    design = np.stack([times**power for power in range(degree + 1)], axis=1)
    coef_x, *_ = np.linalg.lstsq(design, positions[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(design, positions[:, 1], rcond=None)
    pred = np.stack([design @ coef_x, design @ coef_y], axis=1)
    rms = float(np.sqrt(np.mean(np.sum((positions - pred) ** 2, axis=1))))
    return rms, pred


def geometric_line_rms(positions: np.ndarray) -> float:
    """RMS perpendicular distance to the best straight line in image space."""

    centred = positions - positions.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    direction = vh[0]
    projected = np.outer(centred @ direction, direction)
    residual = centred - projected
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def classify_trajectory(
    times: np.ndarray,
    positions: np.ndarray,
    min_motion: float,
    linear_abs: float,
    accel_abs: float,
    relative: float,
    curve_abs: float,
    curve_relative: float,
) -> tuple[str, float, float, float]:
    span = float(np.linalg.norm(positions - positions[len(positions) // 2], axis=1).max())
    if span < min_motion:
        return "static_or_too_small", 0.0, 0.0, span

    linear_rms, _ = fit_rms(times, positions, degree=1)
    accel_rms, _ = fit_rms(times, positions, degree=2)
    linear_thr = max(linear_abs, relative * span)
    accel_thr = max(accel_abs, relative * span)

    if linear_rms <= linear_thr:
        label = "constant_velocity"
    elif accel_rms <= accel_thr:
        curve_rms = geometric_line_rms(positions)
        curve_thr = max(curve_abs, curve_relative * span)
        label = "straight_acceleration" if curve_rms <= curve_thr else "curved_acceleration"
    else:
        label = "other"
    return label, linear_rms, accel_rms, span


def gradient_mask(rgb: np.ndarray, percentile: float) -> np.ndarray:
    gray = to_gray_float(rgb)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    threshold = np.percentile(mag, percentile)
    return mag >= threshold


def resize_flow_to(flow: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a pixel-unit flow field and scale vector components."""

    src_h, src_w = flow.shape[:2]
    if (src_h, src_w) == (height, width):
        return flow
    scale_x = width / float(src_w)
    scale_y = height / float(src_h)
    resized = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= scale_x
    resized[..., 1] *= scale_y
    return resized.astype(np.float32)


def load_memfof_flows(frame_paths: list[Path], height: int, width: int) -> list[np.ndarray] | None:
    center_path = frame_paths[len(frame_paths) // 2]
    flow_paths = [
        memfof_flow_paths_for_center(center_path)[0],
        memfof_flow_paths_for_center(center_path)[1],
        None,
        memfof_flow_paths_for_center(center_path)[2],
        memfof_flow_paths_for_center(center_path)[3],
    ]
    if any(path is not None and not path.exists() for path in flow_paths):
        return None

    flows: list[np.ndarray] = []
    for path in flow_paths:
        if path is None:
            flows.append(np.zeros((height, width, 2), dtype=np.float32))
        else:
            flows.append(resize_flow_to(np.load(path).astype(np.float32), height, width))
    return flows


def sample_records_for_window(
    frame_paths: list[Path],
    max_side: int,
    stride: int,
    method: str,
    args: argparse.Namespace,
) -> tuple[list[TrajectoryRecord], np.ndarray]:
    center_idx = len(frame_paths) // 2
    center = read_rgb(frame_paths[center_idx], max_side=max_side)
    height, width = center.shape[:2]
    times = np.arange(-center_idx, center_idx + 1, dtype=np.float32)

    if args.flow_source == "memfof":
        loaded_flows = load_memfof_flows(frame_paths, height, width)
        if loaded_flows is None:
            return [], center
        flows = loaded_flows
    else:
        frames = [read_rgb(path, max_side=max_side) for path in frame_paths]
        flows: list[np.ndarray] = []
        for idx, frame in enumerate(frames):
            if idx == center_idx:
                flows.append(np.zeros((height, width, 2), dtype=np.float32))
            else:
                flows.append(dense_flow(center, frame, method=method))

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    texture = gradient_mask(center, args.texture_percentile)
    records: list[TrajectoryRecord] = []

    for y in range(stride // 2, height, stride):
        for x in range(stride // 2, width, stride):
            if not texture[y, x]:
                continue
            positions = np.stack(
                [
                    np.array([xx[y, x] + flow[y, x, 0], yy[y, x] + flow[y, x, 1]], dtype=np.float32)
                    for flow in flows
                ],
                axis=0,
            )
            label, linear_rms, accel_rms, span = classify_trajectory(
                times,
                positions,
                min_motion=args.min_motion,
                linear_abs=args.linear_abs_thresh,
                accel_abs=args.accel_abs_thresh,
                relative=args.relative_thresh,
                curve_abs=args.curve_abs_thresh,
                curve_relative=args.curve_relative_thresh,
            )
            records.append(
                TrajectoryRecord(
                    label=label,
                    positions=positions,
                    times=times,
                    linear_rms=linear_rms,
                    accel_rms=accel_rms,
                    span=span,
                    centre_xy=(float(x), float(y)),
                    frame_path=str(frame_paths[center_idx]),
                )
            )
    return records, center


def plot_class_histogram(counts: dict[str, int], output: Path) -> None:
    labels = [
        "constant_velocity",
        "straight_acceleration",
        "curved_acceleration",
        "other",
        "static_or_too_small",
    ]
    values = [counts.get(label, 0) for label in labels]
    total = max(sum(values), 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=160)
    colors = ["#22c55e", "#f59e0b", "#8b5cf6", "#ef4444", "#94a3b8"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("sampled pixel trajectories")
    ax.set_title("GoPro proxy trajectory classes")
    ax.tick_params(axis="x", rotation=15)
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.02, f"{value / total * 100:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_trajectory_examples(records: list[TrajectoryRecord], output: Path, max_per_class: int = 14) -> None:
    labels = ["constant_velocity", "straight_acceleration", "curved_acceleration", "other"]
    fig = plt.figure(figsize=(14, 4.5), dpi=160)
    for panel_idx, label in enumerate(labels, start=1):
        ax = fig.add_subplot(1, 4, panel_idx, projection="3d")
        chosen = [record for record in records if record.label == label]
        chosen = sorted(chosen, key=lambda r: r.span, reverse=True)[:max_per_class]
        for record in chosen:
            pos = record.positions
            ax.plot(pos[:, 0], pos[:, 1], record.times, linewidth=1.0, alpha=0.8)
            ax.scatter(pos[len(pos) // 2, 0], pos[len(pos) // 2, 1], record.times[len(pos) // 2], s=5)
        ax.set_title(f"{label}\n(n={len([r for r in records if r.label == label])})", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("frame t")
        ax.view_init(elev=30, azim=-62)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def normalise_for_average(record: TrajectoryRecord) -> np.ndarray:
    """Return five 3D points [x, y, t] in a common local coordinate frame."""

    positions = record.positions.astype(np.float32)
    centre_idx = len(positions) // 2
    local = positions - positions[centre_idx : centre_idx + 1]

    direction = local[-1] - local[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        centred = positions - positions.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        direction = vh[0]
        norm = float(np.linalg.norm(direction))

    if norm >= 1e-6:
        direction = direction / norm
        if direction[0] < 0:
            direction = -direction
        perp = np.array([-direction[1], direction[0]], dtype=np.float32)
        local = np.stack([local @ direction, local @ perp], axis=1)

    scale = max(float(np.linalg.norm(local, axis=1).max()), 1e-6)
    local = local / scale
    return np.column_stack([local, record.times]).astype(np.float32)


def mean_trajectories(records: list[TrajectoryRecord]) -> dict[str, dict[str, object]]:
    labels = ["constant_velocity", "straight_acceleration", "curved_acceleration", "other"]
    result: dict[str, dict[str, object]] = {}
    for label in labels:
        class_records = [record for record in records if record.label == label]
        if not class_records:
            continue
        points = np.stack([normalise_for_average(record) for record in class_records], axis=0)
        result[label] = {
            "n": len(class_records),
            "mean_points_xyz": points.mean(axis=0).tolist(),
            "std_points_xyz": points.std(axis=0).tolist(),
        }
    return result


def plot_mean_trajectories(records: list[TrajectoryRecord], output: Path) -> None:
    labels = ["constant_velocity", "straight_acceleration", "curved_acceleration", "other"]
    colors = {
        "constant_velocity": "#22c55e",
        "straight_acceleration": "#f59e0b",
        "curved_acceleration": "#8b5cf6",
        "other": "#ef4444",
    }
    fig = plt.figure(figsize=(14, 4.5), dpi=160)
    for panel_idx, label in enumerate(labels, start=1):
        ax = fig.add_subplot(1, 4, panel_idx, projection="3d")
        class_records = [record for record in records if record.label == label]
        if class_records:
            points = np.stack([normalise_for_average(record) for record in class_records], axis=0)
            mean = points.mean(axis=0)
            std = points.std(axis=0)
            for sample in points[: min(80, len(points))]:
                ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], color=colors[label], linewidth=0.45, alpha=0.08)
            ax.plot(mean[:, 0], mean[:, 1], mean[:, 2], color=colors[label], linewidth=3.0)
            ax.scatter(mean[:, 0], mean[:, 1], mean[:, 2], color=colors[label], s=22)
            ax.plot(
                mean[:, 0],
                mean[:, 1] + std[:, 1],
                mean[:, 2],
                color=colors[label],
                linewidth=1.0,
                linestyle="--",
                alpha=0.55,
            )
            ax.plot(
                mean[:, 0],
                mean[:, 1] - std[:, 1],
                mean[:, 2],
                color=colors[label],
                linewidth=1.0,
                linestyle="--",
                alpha=0.55,
            )
        ax.set_title(f"{label}\n(n={len(class_records)})", fontsize=9)
        ax.set_xlabel("aligned x / span")
        ax.set_ylabel("lateral y / span")
        ax.set_zlabel("frame t")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.75, 0.75)
        ax.set_zlim(-2.1, 2.1)
        ax.view_init(elev=30, azim=-62)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def make_overlay(center: np.ndarray, records: list[TrajectoryRecord], output: Path, max_paths: int = 180) -> None:
    image = Image.fromarray(center).convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = {
        "constant_velocity": (34, 197, 94),
        "straight_acceleration": (245, 158, 11),
        "curved_acceleration": (139, 92, 246),
        "other": (239, 68, 68),
        "static_or_too_small": (148, 163, 184),
    }
    moving = [record for record in records if record.label != "static_or_too_small"]
    moving = sorted(moving, key=lambda r: r.span, reverse=True)[:max_paths]
    for record in moving:
        pts = [(float(x), float(y)) for x, y in record.positions]
        color = palette.get(record.label, (255, 255, 255))
        draw.line(pts, fill=color, width=2)
        draw.ellipse((pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2), fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/GoPro"))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--out", type=Path, default=Path("outputs/gopro_proxy_trajectory_classes"))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--flow-source", choices=["opencv", "memfof"], default="opencv")
    parser.add_argument("--flow-method", choices=["dis", "farneback"], default="dis")
    parser.add_argument("--min-motion", type=float, default=1.5)
    parser.add_argument("--linear-abs-thresh", type=float, default=0.75)
    parser.add_argument("--accel-abs-thresh", type=float, default=0.75)
    parser.add_argument("--relative-thresh", type=float, default=0.08)
    parser.add_argument("--curve-abs-thresh", type=float, default=0.45)
    parser.add_argument("--curve-relative-thresh", type=float, default=0.04)
    parser.add_argument("--texture-percentile", type=float, default=55.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    windows = find_windows(
        args.data_root,
        args.split,
        args.windows,
        radius=2,
        require_memfof=args.flow_source == "memfof",
    )
    if not windows:
        raise RuntimeError(f"No five-frame windows found under {args.data_root / args.split}")

    all_records: list[TrajectoryRecord] = []
    first_center = None
    first_records: list[TrajectoryRecord] = []
    processed_windows = 0
    skipped_windows_missing_flow = 0
    for window_idx, frame_paths in enumerate(windows):
        records, center = sample_records_for_window(
            frame_paths,
            max_side=args.max_side,
            stride=args.stride,
            method=args.flow_method,
            args=args,
        )
        if args.flow_source == "memfof" and not records:
            skipped_windows_missing_flow += 1
            continue
        processed_windows += 1
        if window_idx == 0:
            first_center = center
            first_records = records
        all_records.extend(records)

    counts: dict[str, int] = {}
    for record in all_records:
        counts[record.label] = counts.get(record.label, 0) + 1

    plot_class_histogram(counts, args.out / "class_histogram.png")
    plot_trajectory_examples(all_records, args.out / "trajectory_examples_3d.png")
    plot_mean_trajectories(all_records, args.out / "mean_trajectory_3d.png")
    if first_center is not None:
        make_overlay(first_center, first_records, args.out / "first_window_overlay.png")
    mean_report = mean_trajectories(all_records)
    (args.out / "mean_trajectories.json").write_text(json.dumps(mean_report, indent=2), encoding="utf-8")

    total = max(sum(counts.values()), 1)
    report = {
        "data_root": str(args.data_root),
        "split": args.split,
        "windows": len(windows),
        "processed_windows": processed_windows,
        "skipped_windows_missing_flow": skipped_windows_missing_flow,
        "sampled_trajectories": len(all_records),
        "thresholds": {
            "min_motion": args.min_motion,
            "linear_abs_thresh": args.linear_abs_thresh,
            "accel_abs_thresh": args.accel_abs_thresh,
            "relative_thresh": args.relative_thresh,
            "curve_abs_thresh": args.curve_abs_thresh,
            "curve_relative_thresh": args.curve_relative_thresh,
            "texture_percentile": args.texture_percentile,
            "stride": args.stride,
            "flow_source": args.flow_source,
            "flow_method": args.flow_method,
        },
        "counts": counts,
        "percentages": {key: value / total * 100.0 for key, value in counts.items()},
        "outputs": {
            "class_histogram": str(args.out / "class_histogram.png"),
            "trajectory_examples_3d": str(args.out / "trajectory_examples_3d.png"),
            "mean_trajectory_3d": str(args.out / "mean_trajectory_3d.png"),
            "mean_trajectories": str(args.out / "mean_trajectories.json"),
            "first_window_overlay": str(args.out / "first_window_overlay.png"),
        },
        "caveat": "Proxy trajectories estimated from dense optical flow between GoPro sharp frames; not physical exposure GT.",
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
