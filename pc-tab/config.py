"""SIN3D++ Configuration: Control-space parameter sampling.

Conventions: u = reference coords, disp(u,t) = pixel motion, t=0 = center.
RS: t_eff = t + dt(row). Shutter weights sum ~ 1.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class SIN3DConfig:
    """Control-space configuration for SIN3D++ blur synthesis.
    
    Defines 15+ DOF for controlling blur characteristics:
    - Camera motion (translation, rotation, jerk, rolling shutter)
    - Object motion (per-group scale, direction, non-rigid noise)
    - Temporal (shutter length, profile shape)
    - Visibility (occlusion softness, disocclusion)
    - ISP (noise, motion-dependent sharpening)
    
    P2 (optional): camera_model, use_depth_for_camera, intrinsics.
    """
    
    # Camera motion parameters
    camera_translation_scale: Tuple[float, float] = (0.5, 2.0)  # [min, max]
    camera_rotation_scale: Tuple[float, float] = (0.0, 0.1)  # radians
    camera_jerk: Tuple[float, float] = (0.0, 0.05)  # curvature/jerk strength
    rolling_shutter_strength: Tuple[float, float] = (0.0, 0.3)  # row delay factor
    
    # Object motion (per-group)
    object_scale_range: Tuple[float, float] = (0.3, 3.0)
    object_direction_range: Tuple[float, float] = (-180.0, 180.0)  # degrees
    object_nonrigid_noise: Tuple[float, float] = (0.0, 0.1)  # stochastic component
    
    # Temporal parameters
    shutter_length: Tuple[float, float] = (0.8, 1.2)  # relative to exposure
    shutter_profile: str = "box"  # "box", "triangle", "skewed", "gaussian"
    shutter_skew: Tuple[float, float] = (0.3, 0.7)  # for skewed profile
    num_subframes: int = 16  # integration samples
    
    # Visibility parameters
    occlusion_softness: Tuple[float, float] = (0.0, 0.1)  # soft edge width
    disocclusion_hallucination: bool = False  # fill disoccluded regions
    
    # ISP parameters (P5: noise in linear/raw-like; sharpening post-ISP)
    noise_level: Tuple[float, float] = (0.0, 0.02)  # sensor noise std (Gaussian part)
    noise_poisson_scale: Tuple[float, float] = (0.0, 0.01)  # sqrt(L)*scale Poisson-like
    motion_sharpening: Tuple[float, float] = (0.0, 0.1)  # post-ISP sharpening
    
    # P5: Exposure gain (physical, no post-hoc rescaling). Scale radiance in linear.
    exposure_gain: Tuple[float, float] = (1.0, 1.0)  # (min, max); 1.0 = no change
    
    # Depth-dependent scaling
    depth_parallax_scale: Tuple[float, float] = (0.5, 2.0)
    
    # P2: Camera model (SE(3)+depth by default; homography remains fallback)
    camera_model: str = "se3"  # "se3" | "homography"
    use_depth_for_camera: bool = True
    intrinsics: Optional[Tuple[float, float, float, float]] = None  # (fx, fy, cx, cy); None -> approx
    se3_sample_stride: int = 8
    se3_ransac_threshold: float = 3.0
    
    # RS convention: y0 for dt(y) = rs_factor * shutter_length * (y - y0) / H
    rs_y0: float = 0.0  # 0 = top; use 0.5 for center (H/2)
    
    # P3: Bounded non-rigid residual (||d_nr|| <= eps)
    object_residual_max_norm: float = 2.0  # pixels

    # Fix (2): True z-buffer via forward-splat zmin (many-to-one). Default False = min-over-time.
    use_forward_splat_visibility: bool = True

    # Min visibility floor to avoid black borders at depth edges (bilinear warp mixes fg/bg → thin band vis≈0). Default 0.12.
    visibility_floor: float = 0.12

    # Priority A: shutter integration. True = only warp(center, disp(t)) → proper smear; False = temporal blend (ghosting).
    shutter_integration_center_only: bool = True

    # P6: Trajectory profile (constant | acceleration | smooth_walk); invertible.
    trajectory_profile: str = "constant"  # "constant" | "acceleration" | "smooth_walk"
    camera_acceleration: Tuple[float, float] = (0.0, 0.05)  # a in disp = v*t + 0.5*a*t^2
    lateral_acceleration: Tuple[float, float] = (0.0, 0.0)  # perpendicular acceleration for curved paths
    smooth_walk_jerk: Tuple[float, float] = (0.0, 0.02)  # jerk limit for integrated noise

    device: str = "cuda"
    
    def sample(self) -> 'SIN3DParams':
        """Sample a random set of control parameters."""
        return SIN3DParams(
            camera_translation_scale=np.random.uniform(*self.camera_translation_scale),
            camera_rotation_scale=np.random.uniform(*self.camera_rotation_scale),
            camera_jerk=np.random.uniform(*self.camera_jerk),
            rolling_shutter_strength=np.random.uniform(*self.rolling_shutter_strength),
            object_scale=np.random.uniform(*self.object_scale_range),
            object_direction=np.random.uniform(*self.object_direction_range),
            object_nonrigid_noise=np.random.uniform(*self.object_nonrigid_noise),
            shutter_length=np.random.uniform(*self.shutter_length),
            shutter_profile=self.shutter_profile,
            shutter_skew=np.random.uniform(*self.shutter_skew),
            num_subframes=self.num_subframes,
            occlusion_softness=np.random.uniform(*self.occlusion_softness),
            disocclusion_hallucination=self.disocclusion_hallucination,
            noise_level=np.random.uniform(*self.noise_level),
            noise_poisson_scale=np.random.uniform(*self.noise_poisson_scale),
            motion_sharpening=np.random.uniform(*self.motion_sharpening),
            exposure_gain=np.random.uniform(*self.exposure_gain),
            depth_parallax_scale=np.random.uniform(*self.depth_parallax_scale),
            trajectory_profile=self.trajectory_profile,
            camera_acceleration=np.random.uniform(*self.camera_acceleration),
            lateral_acceleration=np.random.uniform(*self.lateral_acceleration),
            smooth_walk_jerk=np.random.uniform(*self.smooth_walk_jerk),
            device=self.device,
        )


@dataclass
class SIN3DParams:
    """Sampled control parameters for a single blur synthesis."""
    camera_translation_scale: float
    camera_rotation_scale: float
    camera_jerk: float
    rolling_shutter_strength: float
    object_scale: float
    object_direction: float  # degrees
    object_nonrigid_noise: float
    shutter_length: float
    shutter_profile: str
    shutter_skew: float
    num_subframes: int
    occlusion_softness: float
    disocclusion_hallucination: bool
    noise_level: float
    noise_poisson_scale: float
    motion_sharpening: float
    exposure_gain: float
    depth_parallax_scale: float
    trajectory_profile: str
    camera_acceleration: float
    lateral_acceleration: float
    smooth_walk_jerk: float
    device: str
    
    def get_time_grid(self):
        """Get normalized time grid [-1, 1] for shutter integration.
        
        Returns:
            If box profile: torch.Tensor [num_subframes]
            Otherwise: tuple (t, weights) where t is [num_subframes] and weights is [num_subframes]
        """
        device = torch.device(self.device)
        if self.shutter_profile == "box":
            return torch.linspace(-1, 1, self.num_subframes, device=device)
        elif self.shutter_profile == "triangle":
            # Weighted towards center
            t = torch.linspace(-1, 1, self.num_subframes, device=device)
            weights = 1.0 - torch.abs(t)
            weights = weights / weights.sum()
            return (t, weights)
        elif self.shutter_profile == "skewed":
            # Skewed towards one end
            t = torch.linspace(-1, 1, self.num_subframes, device=device)
            skew_factor = self.shutter_skew
            if skew_factor > 0.5:
                weights = torch.exp(-skew_factor * (t + 1) ** 2)
            else:
                weights = torch.exp(-(1 - skew_factor) * (t - 1) ** 2)
            weights = weights / weights.sum()
            return (t, weights)
        elif self.shutter_profile == "gaussian":
            t = torch.linspace(-1, 1, self.num_subframes, device=device)
            weights = torch.exp(-0.5 * (t / 0.3) ** 2)
            weights = weights / weights.sum()
            return (t, weights)
        else:
            # Default: box
            return torch.linspace(-1, 1, self.num_subframes, device=device)
