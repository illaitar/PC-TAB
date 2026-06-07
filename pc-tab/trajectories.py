"""SIN3D++ Trajectory Field Builder: Camera, Object, and Stochastic Motion.

P1: Rolling shutter as TIME-WARP (not displacement scaling).
P3: Symmetric object velocity v = 0.5*(res_fwd - res_bwd), layered + bounded residual.
P6: Trajectory profiles constant | acceleration | smooth_walk (invertible).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def get_rs_time_offset_map(
    H: int,
    W: int,
    rs_factor: float,
    shutter_length: float,
    y0: float = 0.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Rolling-shutter time offset per row: dt(y) = rs_factor * shutter_length * (y - y0) / H.

    Output [1, 1, H, 1] broadcastable to [B, N, H, W].
    y0=0 -> top; y0=0.5 -> center (H/2).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32
    y = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    y0_px = y0 * H
    y_norm = (y - y0_px) / max(H, 1)
    dt = rs_factor * shutter_length * y_norm
    return dt  # [1, 1, H, 1]


def _smooth_walk_path(
    N: int,
    jerk: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """P6: Integrated noise path [N] with jerk limit. Invertible (smooth, no i.i.d. per subframe)."""
    if jerk <= 0 or N < 2:
        return torch.linspace(-1.0, 1.0, N, device=device, dtype=dtype)
    dt = 2.0 / max(N - 1, 1)
    j = jerk * (2 * torch.rand(N, device=device, dtype=dtype) - 1)
    a = torch.zeros(N, device=device, dtype=dtype)
    for i in range(1, N):
        a[i] = (a[i - 1] + j[i] * dt).clamp(-1.0, 1.0)
    v = torch.zeros(N, device=device, dtype=dtype)
    for i in range(1, N):
        v[i] = v[i - 1] + a[i] * dt
    p = torch.cumsum(v * dt, dim=0)
    p = p - p[N // 2]
    m = p.abs().max().clamp(min=1e-6)
    return (p / m).to(dtype)


def _perpendicular(flow: torch.Tensor) -> torch.Tensor:
    """Perpendicular 2D direction: (x, y) -> (-y, x)."""
    return torch.stack([-flow[..., 1], flow[..., 0]], dim=-1)


def rotate_vectors_2d(vectors: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate 2D vectors by angle in degrees.
    
    Args:
        vectors: [..., 2] (dx, dy)
        angle_deg: rotation angle in degrees
        
    Returns:
        rotated: [..., 2]
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rot_matrix = torch.tensor(
        [[cos_a, -sin_a], [sin_a, cos_a]],
        device=vectors.device,
        dtype=vectors.dtype
    )
    
    # Apply rotation: [..., 2] @ [2, 2] -> [..., 2]
    if vectors.dim() == 4:  # [B, H, W, 2]
        rotated = torch.einsum('bhwc,cd->bhwd', vectors, rot_matrix)
    elif vectors.dim() == 3:  # [H, W, 2]
        rotated = torch.einsum('hwc,cd->hwd', vectors, rot_matrix)
    else:
        # Generic: last dim is 2
        shape = vectors.shape
        flat = vectors.view(-1, 2)
        rotated_flat = torch.matmul(flat, rot_matrix)
        rotated = rotated_flat.view(*shape)
    
    return rotated


def camera_motion(
    depth: torch.Tensor,  # [B, 1, H, W]
    flow_fwd: torch.Tensor,  # [B, 2, H, W]
    flow_bwd: torch.Tensor,  # [B, 2, H, W]
    params,
    num_subframes: int,
    camera_flow_fwd: Optional[torch.Tensor] = None,  # [B, 2, H, W] pre-extracted camera motion
    camera_flow_bwd: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build depth-aware camera motion trajectory.
    
    P1: Rolling shutter implemented as TIME-WARP (not displacement scaling).
    
    Args:
        depth: [B, 1, H, W] depth map
        flow_fwd: [B, 2, H, W] forward flow (used if camera_flow not provided)
        flow_bwd: [B, 2, H, W] backward flow (used if camera_flow not provided)
        params: SIN3DParams
        num_subframes: number of time samples
        camera_flow_fwd: Optional pre-extracted camera motion (forward)
        camera_flow_bwd: Optional pre-extracted camera motion (backward)
        
    Returns:
        cam_traj: [B, num_subframes, H, W, 2] camera displacement field
    """
    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    
    # Use pre-extracted camera motion if provided, otherwise extract from flow
    if camera_flow_fwd is not None and camera_flow_bwd is not None:
        # Use provided camera motion directly
        base_flow_fwd = camera_flow_fwd.permute(0, 2, 3, 1)  # [B, H, W, 2]
        base_flow_bwd = camera_flow_bwd.permute(0, 2, 3, 1)
        # For trajectory, we need symmetric representation
        base_flow = 0.5 * (base_flow_fwd - base_flow_bwd)  # [B, H, W, 2]
    else:
        # Extract from flow (original method)
        base_flow = 0.5 * (flow_fwd - flow_bwd)  # [B, 2, H, W]
        base_flow = base_flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    
    # SE(3)+depth camera flow is already depth-dependent. Homography/fallback
    # flows are 2D approximations, so do not invent extra inverse-depth parallax.
    z_scale = torch.ones(B, H, W, 1, device=depth.device, dtype=depth.dtype)
    cam_scale = params.camera_translation_scale * params.depth_parallax_scale
    scaled_flow = base_flow * cam_scale * z_scale  # [B, H, W, 2]
    
    # Time grid for shutter integration (t in [-1, 1], center = 0)
    time_grid_result = params.get_time_grid()
    if isinstance(time_grid_result, tuple):
        time_grid, weights = time_grid_result
    else:
        time_grid = time_grid_result
        weights = torch.ones_like(time_grid) / len(time_grid)
    
    rs_y0 = getattr(params, "rs_y0", 0.0)
    rs_map = get_rs_time_offset_map(
        H, W,
        params.rolling_shutter_strength,
        params.shutter_length,
        y0=rs_y0,
        device=device,
        dtype=dtype,
    )
    row_time_offset = rs_map.view(1, H, 1)

    profile = getattr(params, "trajectory_profile", "constant")
    smooth_path = None
    if profile == "smooth_walk":
        smooth_path = _smooth_walk_path(
            num_subframes,
            getattr(params, "smooth_walk_jerk", 0.0),
            device,
            dtype,
        )

    traj = []
    for i, t in enumerate(time_grid):
        t_scaled = t * params.shutter_length
        if profile == "constant" and params.camera_jerk > 0:
            t_warped = t_scaled * (1.0 + params.camera_jerk * t_scaled ** 2)
        else:
            t_warped = t_scaled
        t_eff = (t_warped + row_time_offset).expand(B, H, W)

        if profile == "constant":
            factor = t_eff
        elif profile == "acceleration":
            a = getattr(params, "camera_acceleration", 0.0)
            factor = t_eff + 0.5 * a * (t_eff ** 2)
        elif profile == "smooth_walk" and smooth_path is not None:
            factor = smooth_path[i].view(1, 1, 1).expand(B, H, W).to(device)
        else:
            factor = t_eff

        disp = scaled_flow * factor.unsqueeze(-1)
        lateral = getattr(params, "lateral_acceleration", 0.0)
        if lateral != 0:
            disp = disp + 0.5 * lateral * (t_eff ** 2).unsqueeze(-1) * _perpendicular(scaled_flow)
        traj.append(disp)

    return torch.stack(traj, dim=1)


def object_motion(
    masks: Optional[torch.Tensor],  # [B, K, H, W] optional object masks
    flow_fwd: torch.Tensor,  # [B, 2, H, W]
    flow_bwd: torch.Tensor,  # [B, 2, H, W]
    params,
    num_subframes: int,
) -> torch.Tensor:
    """Build per-object independent motion trajectory.
    
    Args:
        masks: [B, K, H, W] object masks (optional)
        flow_fwd: [B, 2, H, W] full flow (or residual if camera extracted)
        flow_bwd: [B, 2, H, W] full flow (or residual if camera extracted)
        params: SIN3DParams (may contain _camera_flow_fwd/_bwd for residual)
        num_subframes: number of time samples
        
    Returns:
        obj_traj: [B, num_subframes, H, W, 2]
    """
    B, _, H, W = flow_fwd.shape
    device = flow_fwd.device
    dtype = flow_fwd.dtype
    
    if masks is None or masks.shape[1] == 0:
        # No object masks: return zero motion
        return torch.zeros(B, num_subframes, H, W, 2, device=device, dtype=dtype)
    
    K = masks.shape[1]
    
    # Debug: check if masks have content
    mask_sum = masks.sum().item()
    if mask_sum < 100:
        # Masks are mostly empty, return zero motion
        return torch.zeros(B, num_subframes, H, W, 2, device=device, dtype=dtype)
    
    # P3: Symmetric object velocity v = 0.5 * (res_fwd - res_bwd)
    camera_flow_fwd = getattr(params, '_camera_flow_fwd', None)
    camera_flow_bwd = getattr(params, '_camera_flow_bwd', None)
    if camera_flow_fwd is not None and camera_flow_bwd is not None:
        res_fwd = flow_fwd - camera_flow_fwd  # [B, 2, H, W]
        res_bwd = flow_bwd - camera_flow_bwd
        v = 0.5 * (res_fwd - res_bwd)  # symmetric velocity
        base_flow = v.permute(0, 2, 3, 1)  # [B, H, W, 2]
    else:
        # Priority B: symmetric velocity even when camera not extracted (not only flow_fwd).
        v = 0.5 * (flow_fwd - flow_bwd)
        base_flow = v.permute(0, 2, 3, 1)  # [B, H, W, 2]
    
    time_grid_result = params.get_time_grid()
    if isinstance(time_grid_result, tuple):
        time_grid, _ = time_grid_result
    else:
        time_grid = time_grid_result

    rs_y0 = getattr(params, "rs_y0", 0.0)
    rs_map = get_rs_time_offset_map(H, W, params.rolling_shutter_strength, params.shutter_length, y0=rs_y0, device=device, dtype=dtype)
    row_dt = rs_map.view(1, H, 1)

    profile = getattr(params, "trajectory_profile", "constant")
    smooth_path = None
    if profile == "smooth_walk":
        smooth_path = _smooth_walk_path(
            num_subframes,
            getattr(params, "smooth_walk_jerk", 0.0),
            device,
            dtype,
        )

    obj_scale = params.object_scale
    if params.object_direction != 0.0:
        base_flow = rotate_vectors_2d(base_flow, params.object_direction)
    scaled_flow = base_flow * obj_scale

    traj = torch.zeros(B, num_subframes, H, W, 2, device=device, dtype=dtype)
    for k in range(K):
        mask = masks[:, k : k + 1]
        mask_expanded = mask.permute(0, 2, 3, 1)

        mask_traj = []
        for i, t in enumerate(time_grid):
            t_scaled = t * params.shutter_length
            t_eff = (t_scaled + row_dt).expand(B, H, W)

            if profile == "constant":
                factor = t_eff
            elif profile == "acceleration":
                a = getattr(params, "camera_acceleration", 0.0)
                factor = t_eff + 0.5 * a * (t_eff ** 2)
            elif profile == "smooth_walk" and smooth_path is not None:
                factor = smooth_path[i].view(1, 1, 1).expand(B, H, W).to(device)
            else:
                factor = t_eff

            disp = scaled_flow * factor.unsqueeze(-1)
            lateral = getattr(params, "lateral_acceleration", 0.0)
            if lateral != 0:
                disp = disp + 0.5 * lateral * (t_eff ** 2).unsqueeze(-1) * _perpendicular(scaled_flow)
            mask_disp = disp * mask_expanded
            mask_traj.append(mask_disp)

        mask_traj = torch.stack(mask_traj, dim=1)
        traj = traj + mask_traj
    
    # P3: Bounded non-rigid residual. P6: No i.i.d. per subframe when smooth_walk (breaks invertibility).
    profile = getattr(params, "trajectory_profile", "constant")
    if params.object_nonrigid_noise > 0 and profile != "smooth_walk":
        noise = torch.randn(B, num_subframes, H, W, 2, device=device, dtype=dtype)
        noise = noise * params.object_nonrigid_noise
        eps = getattr(params, "object_residual_max_norm", 2.0)
        n = torch.linalg.norm(noise, dim=-1, keepdim=True).clamp(min=1e-6)
        noise = noise * (n.clamp(max=eps) / n)
        traj = traj + noise

    return traj


def build_piecewise_trajectory(
    cam_traj: torch.Tensor,  # [B, N, H, W, 2]
    obj_traj: torch.Tensor,  # [B, N, H, W, 2]
) -> torch.Tensor:
    """Combine camera and object trajectories.
    
    Args:
        cam_traj: [B, N, H, W, 2] camera trajectory
        obj_traj: [B, N, H, W, 2] object trajectory
        
    Returns:
        traj: [B, N, H, W, 2] combined trajectory
    """
    return cam_traj + obj_traj
