"""Break down the residual `other` trajectory class into diagnostic subtypes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_analyzer():
    path = Path(__file__).with_name("analyze_gopro_proxy_trajectories.py")
    spec = importlib.util.spec_from_file_location("gopro_traj_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gopro_traj_analyzer"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def local_flow_jump(flows: list[np.ndarray], y: int, x: int, radius: int = 2) -> float:
    jumps: list[float] = []
    for flow in flows:
        if np.allclose(flow[y, x], 0.0):
            continue
        y0 = max(0, y - radius)
        y1 = min(flow.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(flow.shape[1], x + radius + 1)
        patch = flow[y0:y1, x0:x1].reshape(-1, 2)
        centre = flow[y, x]
        jumps.append(float(np.median(np.linalg.norm(patch - centre, axis=1))))
    return float(np.max(jumps)) if jumps else 0.0


def classify_other_subtype(
    analyzer,
    label: str,
    positions: np.ndarray,
    times: np.ndarray,
    span: float,
    accel_rms: float,
    curve_rms: float,
    flow_jump: float,
    args: argparse.Namespace,
) -> str:
    if label != "other":
        return label

    rec = analyzer.TrajectoryRecord(label, positions, times, 0.0, accel_rms, span, (0.0, 0.0), "")
    local = analyzer.normalise_for_average(rec)
    xseq = local[:, 0]
    monotonic = bool(np.all(np.diff(xseq) >= -args.monotonic_slack) or np.all(np.diff(xseq) <= args.monotonic_slack))

    velocities = np.diff(positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    speed_cv = float(speeds.std() / (speeds.mean() + 1e-6))
    accel_thr = max(args.accel_abs_thresh, args.relative_thresh * span)
    curve_thr = max(args.curve_abs_thresh, args.curve_relative_thresh * span)
    boundary_thr = max(args.boundary_abs_thresh, args.boundary_relative_thresh * span)

    if flow_jump > boundary_thr:
        return "other_boundary_or_occlusion"
    if not monotonic:
        return "other_non_monotonic_inconsistent"
    if curve_rms > args.high_curve_multiplier * curve_thr:
        return "other_high_curvature_nonquadratic"
    if speed_cv > args.speed_cv_thresh and accel_rms > args.high_accel_multiplier * accel_thr:
        return "other_irregular_speed_profile"
    return "other_mild_residual"


def plot_breakdown(counts: dict[str, int], output: Path) -> None:
    labels = [
        "constant_velocity",
        "straight_acceleration",
        "curved_acceleration",
        "other_boundary_or_occlusion",
        "other_non_monotonic_inconsistent",
        "other_high_curvature_nonquadratic",
        "other_irregular_speed_profile",
        "other_mild_residual",
    ]
    values = [counts.get(label, 0) for label in labels]
    colors = ["#22c55e", "#f59e0b", "#8b5cf6", "#0ea5e9", "#ef4444", "#ec4899", "#f97316", "#64748b"]
    total = max(sum(values), 1)
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    ax.bar(labels, values, color=colors)
    ax.set_title("MEMFOF trajectory classes with `other` breakdown")
    ax.set_ylabel("sampled pixel trajectories")
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.02, f"{value / total * 100:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/GoPro"))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--windows", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("outputs/gopro_memfof_other_breakdown"))
    parser.add_argument("--min-motion", type=float, default=1.5)
    parser.add_argument("--linear-abs-thresh", type=float, default=0.75)
    parser.add_argument("--accel-abs-thresh", type=float, default=0.75)
    parser.add_argument("--relative-thresh", type=float, default=0.08)
    parser.add_argument("--curve-abs-thresh", type=float, default=0.45)
    parser.add_argument("--curve-relative-thresh", type=float, default=0.04)
    parser.add_argument("--texture-percentile", type=float, default=55.0)
    parser.add_argument("--monotonic-slack", type=float, default=0.05)
    parser.add_argument("--boundary-abs-thresh", type=float, default=3.0)
    parser.add_argument("--boundary-relative-thresh", type=float, default=0.12)
    parser.add_argument("--high-curve-multiplier", type=float, default=1.8)
    parser.add_argument("--high-accel-multiplier", type=float, default=1.5)
    parser.add_argument("--speed-cv-thresh", type=float, default=0.45)
    args = parser.parse_args()

    analyzer = load_analyzer()
    windows = analyzer.find_windows(args.data_root, args.split, args.windows, radius=2, require_memfof=True)
    args.out.mkdir(parents=True, exist_ok=True)

    times = np.arange(-2, 3, dtype=np.float32)
    counts: dict[str, int] = {}
    metrics: dict[str, list[dict[str, float]]] = {}

    for frame_paths in windows:
        center = analyzer.read_rgb(frame_paths[2], max_side=args.max_side)
        height, width = center.shape[:2]
        flows = analyzer.load_memfof_flows(frame_paths, height, width)
        if flows is None:
            continue

        texture = analyzer.gradient_mask(center, args.texture_percentile)
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        for y in range(args.stride // 2, height, args.stride):
            for x in range(args.stride // 2, width, args.stride):
                if not texture[y, x]:
                    continue
                positions = np.stack(
                    [
                        np.array([xx[y, x] + flow[y, x, 0], yy[y, x] + flow[y, x, 1]], dtype=np.float32)
                        for flow in flows
                    ],
                    axis=0,
                )
                label, linear_rms, accel_rms, span = analyzer.classify_trajectory(
                    times,
                    positions,
                    min_motion=args.min_motion,
                    linear_abs=args.linear_abs_thresh,
                    accel_abs=args.accel_abs_thresh,
                    relative=args.relative_thresh,
                    curve_abs=args.curve_abs_thresh,
                    curve_relative=args.curve_relative_thresh,
                )
                if label == "static_or_too_small":
                    continue
                curve_rms = analyzer.geometric_line_rms(positions)
                flow_jump = local_flow_jump(flows, y, x)
                subtype = classify_other_subtype(
                    analyzer,
                    label,
                    positions,
                    times,
                    span,
                    accel_rms,
                    curve_rms,
                    flow_jump,
                    args,
                )
                counts[subtype] = counts.get(subtype, 0) + 1
                metrics.setdefault(subtype, []).append(
                    {
                        "span": span,
                        "linear_rms": linear_rms,
                        "accel_rms": accel_rms,
                        "curve_rms": curve_rms,
                        "flow_jump": flow_jump,
                    }
                )

    total = max(sum(counts.values()), 1)
    summary_metrics = {}
    for label, rows in metrics.items():
        arr = {key: np.array([row[key] for row in rows], dtype=np.float32) for key in rows[0]}
        summary_metrics[label] = {
            key: {
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
            }
            for key, values in arr.items()
        }

    plot_breakdown(counts, args.out / "other_breakdown_histogram.png")
    report = {
        "split": args.split,
        "ready_memfof_windows": len(windows),
        "sampled_trajectories": sum(counts.values()),
        "counts": counts,
        "percentages": {key: value / total * 100.0 for key, value in counts.items()},
        "metrics": summary_metrics,
        "thresholds": vars(args) | {"out": str(args.out), "data_root": str(args.data_root)},
        "outputs": {
            "histogram": str(args.out / "other_breakdown_histogram.png"),
            "summary": str(args.out / "other_breakdown_summary.json"),
        },
    }
    (args.out / "other_breakdown_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
