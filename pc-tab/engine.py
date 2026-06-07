"""SIN3D++ Engine: Main synthesis pipeline with physically-correct improvements.

P0: True multi-frame integration
P1: Rolling shutter as time-warp
P3: Depth-ordered visibility
P4: Linear-light integration, no brightness normalization
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Union

from .trajectories import build_piecewise_trajectory, camera_motion, object_motion
from .visibility import compute_visibility
from .shutter import integrate_shutter
from .isp import apply_isp
from .config import SIN3DConfig, SIN3DParams
from .camera_extractor import (
    extract_object_masks_from_residual,
    extract_homography_from_frames,
    homography_to_motion_field,
    pose_to_motion_field,
    estimate_se3_from_flow_depth,
)


class SIN3DEngine:
    """SIN3D++ blur synthesis engine with physically-correct improvements."""
    
    def __init__(
        self,
        cfg: SIN3DConfig,
        residual_threshold: float = 2.0,
        extract_objects: bool = True,
        homography_ransac_threshold: float = 3.0,
        use_linear_light: bool = True,  # P4: integrate in linear light
        use_depth_ordering: bool = True,  # P3: depth-ordered visibility
    ):
        """Initialize SIN3D++ engine.
        
        Args:
            cfg: SIN3DConfig configuration
            residual_threshold: Threshold for extracting object masks from residual flow
            extract_objects: If True, automatically extract object masks from residual flow
            homography_ransac_threshold: RANSAC threshold for homography estimation (pixels)
            use_linear_light: If True, integrate in linear light space (P4)
            use_depth_ordering: If True, use depth-ordered visibility (P3)
        """
        self.cfg = cfg
        self.residual_threshold = residual_threshold
        self.extract_objects = extract_objects
        self.homography_ransac_threshold = homography_ransac_threshold
        self.use_linear_light = use_linear_light
        self.use_depth_ordering = use_depth_ordering
    
    @torch.no_grad()
    def synthesize(
        self,
        sharp_seq: torch.Tensor,  # [T, 3, H, W] or [B, T, 3, H, W]
        depth: torch.Tensor,  # [1, H, W] or [B, 1, H, W]; use depth_da3 when using pose
        flow_fwd: torch.Tensor,  # [2, H, W] or [B, 2, H, W]
        flow_bwd: torch.Tensor,  # [2, H, W] or [B, 2, H, W]
        masks: Optional[torch.Tensor] = None,  # [K, H, W] or [B, K, H, W] optional
        occ_mask: Optional[torch.Tensor] = None,  # [1, H, W] or [B, 1, H, W] optional
        params: Optional[SIN3DParams] = None,
        extrinsics: Optional[Union[torch.Tensor, np.ndarray]] = None,  # [3, 3, 4] prev, center, next (DA3)
        intrinsics: Optional[Union[torch.Tensor, np.ndarray]] = None,  # [3, 3] K
    ) -> Tuple[torch.Tensor, Dict]:
        """Synthesize blur from sharp sequence.
        
        Args:
            sharp_seq: sharp frames [T, 3, H, W] or [B, T, 3, H, W]
            depth: depth map [1, H, W] or [B, 1, H, W]; use depth_da3 when using pose
            flow_fwd: forward flow [2, H, W] or [B, 2, H, W]
            flow_bwd: backward flow [2, H, W] or [B, 2, H, W]
            masks: optional object masks [K, H, W] or [B, K, H, W]
            occ_mask: optional occlusion mask [1, H, W] or [B, 1, H, W]
            params: optional SIN3DParams (if None, samples from config)
            extrinsics: optional [3, 3, 4] DA3 pose (prev, center, next); use when camera_model=se3
            intrinsics: optional [3, 3] K; use when camera_model=se3
            
        Returns:
            blur: [B, 3, H, W] or [3, H, W] synthesized blur
            meta: dict with trajectories, visibility, params
        """
        # Sample parameters if not provided
        if params is None:
            params = self.cfg.sample()
        
        # Normalize device
        device = depth.device if isinstance(depth, torch.Tensor) else torch.device(self.cfg.device)
        params.device = str(device)
        
        # Normalize sharp_seq dimensions
        if sharp_seq.dim() == 4:  # [T, 3, H, W]
            sharp_seq = sharp_seq.unsqueeze(0)  # [1, T, 3, H, W]
            single_batch = True
        else:
            single_batch = False
        
        B, T, C, H, W = sharp_seq.shape
        center_idx = T // 2
        
        # Store center_idx in params for later use
        params._center_idx = center_idx
        
        sharp_prev = sharp_seq[:, center_idx - 1] if center_idx > 0 else sharp_seq[:, center_idx]
        sharp = sharp_seq[:, center_idx]
        sharp_next = sharp_seq[:, center_idx + 1] if center_idx < T - 1 else sharp_seq[:, center_idx]
        
        # Normalize flow dimensions
        if flow_fwd.dim() == 3:
            flow_fwd = flow_fwd.unsqueeze(0)
        if flow_bwd.dim() == 3:
            flow_bwd = flow_bwd.unsqueeze(0)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0) if depth.shape[0] == 1 else depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[1] != 1:
            depth = depth[:, :1]
        
        diagnostics = {}
        camera_model = getattr(self.cfg, "camera_model", "se3")
        if camera_model == "se3" and (extrinsics is None or intrinsics is None) and getattr(self.cfg, "use_depth_for_camera", True):
            try:
                extrinsics, intrinsics, diagnostics = estimate_se3_from_flow_depth(
                    depth=depth,
                    flow_fwd=flow_fwd,
                    flow_bwd=flow_bwd,
                    intrinsics=getattr(self.cfg, "intrinsics", None),
                    stride=getattr(self.cfg, "se3_sample_stride", 8),
                    ransac_threshold=getattr(self.cfg, "se3_ransac_threshold", 3.0),
                )
            except Exception as exc:
                diagnostics = {"se3_estimator": "failed", "se3_error": str(exc)}

        # Extract camera motion: pose (DA3 or robust RGB-D flow PnP) when available, else homography.
        use_pose = camera_model == "se3" and extrinsics is not None and intrinsics is not None
        if use_pose:
            cam_fwd, cam_bwd = pose_to_motion_field(extrinsics, intrinsics, depth, eps=1e-4)
        else:
            try:
                H_prev, _ = extract_homography_from_frames(
                    sharp, sharp_prev,
                    ransac_threshold=self.homography_ransac_threshold
                )
                H_next, _ = extract_homography_from_frames(
                    sharp, sharp_next,
                    ransac_threshold=self.homography_ransac_threshold
                )
                cam_bwd = homography_to_motion_field(H_prev, H, W)
                cam_fwd = homography_to_motion_field(H_next, H, W)
                if cam_fwd.dim() == 3:
                    cam_fwd = cam_fwd.unsqueeze(0)
                    cam_bwd = cam_bwd.unsqueeze(0)
            except Exception:
                cam_fwd = 0.5 * (flow_fwd - flow_bwd)
                cam_bwd = -cam_fwd
        
        # Compute residual flow (object motion) = flow - camera_motion
        res_fwd = flow_fwd - cam_fwd  # [B, 2, H, W]
        res_bwd = flow_bwd - cam_bwd
        
        # Extract object masks from residual flow (if enabled and masks not provided)
        if self.extract_objects and masks is None:
            masks = extract_object_masks_from_residual(
                res_fwd,  # Use forward residual
                threshold=self.residual_threshold,
            )  # [B, K, H, W]
        
        params._camera_flow_fwd = cam_fwd
        params._camera_flow_bwd = cam_bwd
        params._camera_from_pose = use_pose  # Fix (3): z_scale only when SE(3)+depth
        params._camera_model_used = "se3" if use_pose else "homography_or_flow_fallback"
        params.rs_y0 = getattr(self.cfg, "rs_y0", 0.0)
        params.object_residual_max_norm = getattr(self.cfg, "object_residual_max_norm", 2.0)
        
        # 1. Build trajectory field
        sharp_seq_for_traj = sharp_seq.squeeze(0) if single_batch else sharp_seq
        
        # Build camera trajectory (P1: rolling shutter as time-warp)
        cam_traj = camera_motion(
            depth=depth,
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            params=params,
            num_subframes=params.num_subframes,
            camera_flow_fwd=cam_fwd,
            camera_flow_bwd=cam_bwd,
        )  # [B, num_subframes, H, W, 2]
        
        # Build object trajectory
        obj_traj = object_motion(
            masks=masks,
            flow_fwd=flow_fwd,
            flow_bwd=flow_bwd,
            params=params,
            num_subframes=params.num_subframes,
        )  # [B, num_subframes, H, W, 2]
        
        # Combine trajectories
        traj = build_piecewise_trajectory(cam_traj, obj_traj)  # [B, N, H, W, 2]
        
        # 2. Compute visibility/occlusion (P3: depth-ordered; Fix 2: optional forward-splat zmin; visibility_floor avoids black borders at depth edges)
        params.visibility_floor = getattr(self.cfg, "visibility_floor", 0.12)
        visibility = compute_visibility(
            traj=traj,
            depth=depth,
            params=params,
            occ_mask=occ_mask,
            use_depth_ordering=self.use_depth_ordering,
            use_forward_splat_zmin=getattr(self.cfg, "use_forward_splat_visibility", False),  # Fix (B): config flag
        )  # [B, N, H, W]
        
        # 3. Shutter integration. Contract: sharp_seq in [0,1] or [-1,1]; output blur in [0,1] (no extra (blur+1)/2).
        if sharp_seq.min() < 0:
            sharp_seq_norm = (sharp_seq + 1.0) / 2.0
        else:
            sharp_seq_norm = sharp_seq

        return_linear = self.use_linear_light  # P5: hand linear to ISP for noise in linear
        use_center_only = getattr(self.cfg, "shutter_integration_center_only", True)  # Priority A: proper smear
        blur = integrate_shutter(
            sharp_seq=sharp_seq_norm,
            traj=traj,
            visibility=visibility,
            params=params,
            use_linear_light=self.use_linear_light,
            use_zbuffer_no_weight_norm=self.use_depth_ordering,
            return_linear=return_linear,
            use_center_only=use_center_only,
        )

        if blur.dim() == 3:
            blur = blur.unsqueeze(0)

        # 4. ISP: P5 noise in linear (Poisson+Gaussian) -> linear_to_srgb -> post-ISP sharpening
        blur = apply_isp(blur, params, use_linear_input=return_linear)
        
        # Clamp to valid range
        blur = torch.clamp(blur, 0.0, 1.0)
        
        # Restore original dimensions
        if single_batch:
            blur = blur.squeeze(0)  # [3, H, W]
        
        # Prepare metadata (for motion-decomposition viz: cam_traj, obj_traj, flows, masks)
        meta = {
            "traj": traj,
            "visibility": visibility,
            "params": params,
            "camera_model": params._camera_model_used,
            "camera_diagnostics": diagnostics,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "cam_traj": cam_traj,
            "obj_traj": obj_traj,
            "cam_fwd": cam_fwd,
            "cam_bwd": cam_bwd,
            "res_fwd": res_fwd,
            "res_bwd": res_bwd,
            "masks": masks,
        }
        
        return blur, meta
