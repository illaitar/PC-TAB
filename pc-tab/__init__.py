"""SIN3D++: Physically-Correct Blur Synthesis.

- P0: True multi-frame exposure integration; I_src(t_eff).
- P1: Rolling shutter as time-warp; RS affects motion and frame selection.
- P3: Object motion symmetric velocity v=0.5*(res_fwd−res_bwd), layered, bounded residual.
- P4: Soft z-buffer w_vis=exp(-(z−zmin)/tau); linear-light integration; no /weight_sum when z-buffer.
"""

from .engine import SIN3DEngine
from .config import SIN3DConfig, SIN3DParams
from .shutter import integrate_shutter
from .trajectories import camera_motion, object_motion, build_piecewise_trajectory, get_rs_time_offset_map
from .visibility import compute_visibility
from .camera_extractor import (
    extract_homography_from_frames,
    homography_to_motion_field,
    estimate_se3_from_flow_depth,
    make_intrinsics,
    pose_to_motion_field,
    extract_object_masks_from_residual,
)
from .regularizers import (
    cycle_consistency_loss,
    edge_aware_smooth_loss,
    acceleration_prior_loss,
    forward_backward_consistency_loss,
    MotionRegularizers,
)

__all__ = [
    "SIN3DEngine",
    "SIN3DConfig",
    "SIN3DParams",
    "integrate_shutter",
    "camera_motion",
    "object_motion",
    "build_piecewise_trajectory",
    "get_rs_time_offset_map",
    "compute_visibility",
    "extract_homography_from_frames",
    "homography_to_motion_field",
    "estimate_se3_from_flow_depth",
    "make_intrinsics",
    "pose_to_motion_field",
    "extract_object_masks_from_residual",
    "cycle_consistency_loss",
    "edge_aware_smooth_loss",
    "acceleration_prior_loss",
    "forward_backward_consistency_loss",
    "MotionRegularizers",
]
