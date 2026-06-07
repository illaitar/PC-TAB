"""Camera Motion Extraction from Sharp Frames and Object Mask Extraction.

- Extracts camera motion via homography from 3 sharp frames
- Extracts object masks by thresholding residual flow (flow - camera_motion)
- pose_to_motion_field: SE(3) + depth -> camera flow (for DA3 extrinsics + intrinsics)
- estimate_se3_from_flow_depth: robust RGB-D pose from MEMFOF flow + depth
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def extract_homography_from_frames(
    img1: torch.Tensor,  # [B, 3, H, W] or [3, H, W] in [-1, 1] or [0, 1]
    img2: torch.Tensor,  # [B, 3, H, W] or [3, H, W] in [-1, 1] or [0, 1]
    method: str = "cv2",
    ransac_threshold: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.995,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract homography between two images.
    
    Args:
        img1: Reference image [B, 3, H, W] or [3, H, W]
        img2: Target image [B, 3, H, W] or [3, H, W]
        method: "cv2" (OpenCV) or "kornia" (if available)
        ransac_threshold: RANSAC threshold in pixels
        max_iters: Maximum RANSAC iterations
        confidence: Required confidence for RANSAC
        
    Returns:
        H: Homography matrix [B, 3, 3] or [3, 3]
        mask: Inlier mask [B, H, W] or [H, W] (optional, may be None)
    """
    if not HAS_CV2 and method == "cv2":
        raise ImportError("OpenCV (cv2) is required for homography extraction")
    
    # Normalize dimensions
    single_batch = False
    if img1.dim() == 3:  # [3, H, W]
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        single_batch = True
    
    B, C, img_H, img_W = img1.shape
    device = img1.device
    dtype = img1.dtype
    
    # Convert to numpy for OpenCV
    # Normalize from [-1, 1] or [0, 1] to [0, 255] uint8
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    
    if img1_np.min() < 0:  # [-1, 1] range
        img1_np = (img1_np + 1.0) / 2.0
        img2_np = (img2_np + 1.0) / 2.0
    
    img1_np = (img1_np * 255.0).clip(0, 255).astype(np.uint8)
    img2_np = (img2_np * 255.0).clip(0, 255).astype(np.uint8)
    
    # Convert RGB to grayscale for feature matching
    homographies = []
    inlier_masks = []
    
    for b in range(B):
        # Convert to grayscale
        if C == 3:
            gray1 = cv2.cvtColor(img1_np[b].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2_np[b].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1_np[b, 0]
            gray2 = img2_np[b, 0]
        
        # Detect features (SIFT or ORB)
        try:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
        except:
            # Fallback to ORB
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            # Not enough features: return identity
            H_eye = np.eye(3, dtype=np.float32)
            homographies.append(H_eye)
            inlier_masks.append(np.ones((img_H, img_W), dtype=np.float32))
            continue
        
        # Match features
        if des1.dtype == np.uint8:  # ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # SIFT
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            # Not enough matches: return identity
            H_eye = np.eye(3, dtype=np.float32)
            homographies.append(H_eye)
            inlier_masks.append(np.ones((img_H, img_W), dtype=np.float32))
            continue
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography with RANSAC
        H_mat, inlier_mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
        
        if H_mat is None:
            H_mat = np.eye(3, dtype=np.float32)
            inlier_mask = np.ones((len(good_matches),), dtype=np.uint8)
        
        homographies.append(H_mat.astype(np.float32))
        
        # Create inlier mask image (optional, for visualization)
        # We could project points to see which pixels are inliers, but for now just return ones
        inlier_masks.append(np.ones((img_H, img_W), dtype=np.float32))
    
    # Stack homographies
    H_tensor = torch.from_numpy(np.stack(homographies)).to(device=device, dtype=dtype)
    mask_tensor = torch.from_numpy(np.stack(inlier_masks)).to(device=device, dtype=dtype)
    
    if single_batch:
        H_tensor = H_tensor.squeeze(0)
        mask_tensor = mask_tensor.squeeze(0)
    
    return H_tensor, mask_tensor


def homography_to_motion_field(
    H: torch.Tensor,  # [B, 3, 3] or [3, 3] homography matrix
    H_img: int,
    W_img: int,
) -> torch.Tensor:
    """Convert homography matrix to dense motion field.
    
    Args:
        H: Homography matrix [B, 3, 3] or [3, 3]
        H_img: Image height
        W_img: Image width
        
    Returns:
        motion_field: [B, 2, H, W] or [2, H, W] displacement field in pixels
    """
    single_batch = False
    if H.dim() == 2:  # [3, 3]
        H = H.unsqueeze(0)
        single_batch = True
    
    B = H.shape[0]
    device = H.device
    dtype = H.dtype
    
    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.arange(H_img, device=device, dtype=dtype),
        torch.arange(W_img, device=device, dtype=dtype),
        indexing='ij'
    )
    
    # Convert to homogeneous coordinates [B, 3, H*W]
    coords = torch.stack([x.flatten(), y.flatten(), torch.ones(H_img * W_img, device=device, dtype=dtype)], dim=0)
    coords = coords.unsqueeze(0).expand(B, -1, -1)  # [B, 3, H*W]
    
    # Apply homography: H @ coords
    coords_transformed = torch.bmm(H, coords)  # [B, 3, H*W]
    
    # Convert back from homogeneous: divide by z
    z = coords_transformed[:, 2:3, :].clamp(min=1e-6)
    coords_transformed = coords_transformed[:, :2, :] / z  # [B, 2, H*W]
    
    # Reshape to [B, 2, H, W]
    motion_field = coords_transformed.view(B, 2, H_img, W_img)
    
    # Subtract original coordinates to get displacement
    coords_orig = torch.stack([x, y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
    motion_field = motion_field - coords_orig
    
    if single_batch:
        motion_field = motion_field.squeeze(0)
    
    return motion_field


def make_intrinsics(
    H_img: int,
    W_img: int,
    intrinsics: Optional[Union[Tuple[float, float, float, float], torch.Tensor, np.ndarray]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a pinhole intrinsic matrix.

    If no calibration is supplied, use a conservative GoPro-like approximation:
    focal length equals the larger image side and principal point is centered.
    """

    if intrinsics is None:
        fx = fy = float(max(H_img, W_img))
        cx = (W_img - 1.0) * 0.5
        cy = (H_img - 1.0) * 0.5
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=dtype)
    elif isinstance(intrinsics, tuple):
        fx, fy, cx, cy = intrinsics
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=dtype)
    elif isinstance(intrinsics, np.ndarray):
        K = torch.from_numpy(intrinsics).to(dtype=dtype)
    else:
        K = intrinsics.to(dtype=dtype)
    if device is not None:
        K = K.to(device=device)
    return K


def _estimate_relative_pose_pnp(
    flow: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    stride: int = 8,
    ransac_threshold: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 5000,
) -> Tuple[torch.Tensor, int, float]:
    """Estimate center->target camera transform from center depth and target flow."""

    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for SE(3) estimation")

    device = depth.device
    dtype = depth.dtype
    if depth.dim() == 4:
        depth_2d = depth[0, 0]
    elif depth.dim() == 3:
        depth_2d = depth[0]
    else:
        depth_2d = depth

    if flow.dim() == 4:
        flow_hw = flow[0].permute(1, 2, 0)
    elif flow.dim() == 3 and flow.shape[0] == 2:
        flow_hw = flow.permute(1, 2, 0)
    else:
        flow_hw = flow

    H_img, W_img = depth_2d.shape
    yy, xx = torch.meshgrid(
        torch.arange(0, H_img, stride, device=device, dtype=dtype),
        torch.arange(0, W_img, stride, device=device, dtype=dtype),
        indexing="ij",
    )
    yy_i = yy.long().flatten()
    xx_i = xx.long().flatten()
    z = depth_2d[yy_i, xx_i]
    uv_flow = flow_hw[yy_i, xx_i]
    valid = torch.isfinite(z) & torch.isfinite(uv_flow).all(dim=1) & (z > 1e-4)
    if valid.sum() < 12:
        E = torch.eye(3, 4, device=device, dtype=dtype)
        return E, 0, float("inf")

    xx_v = xx_i[valid].to(dtype)
    yy_v = yy_i[valid].to(dtype)
    z_v = z[valid]
    flow_v = uv_flow[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xx_v - cx) / fx * z_v
    Y = (yy_v - cy) / fy * z_v
    object_points = torch.stack([X, Y, z_v], dim=1).detach().cpu().numpy().astype(np.float32)
    image_points = torch.stack([xx_v + flow_v[:, 0], yy_v + flow_v[:, 1]], dim=1).detach().cpu().numpy().astype(np.float32)
    K_np = K.detach().cpu().numpy().astype(np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K_np,
        None,
        iterationsCount=max_iters,
        reprojectionError=ransac_threshold,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 12:
        E = torch.eye(3, 4, device=device, dtype=dtype)
        return E, 0, float("inf")

    # Refine on inliers where available.
    try:
        inlier_idx = inliers.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_idx],
            image_points[inlier_idx],
            K_np,
            None,
            rvec,
            tvec,
        )
    except Exception:
        pass

    R, _ = cv2.Rodrigues(rvec)
    E_np = np.concatenate([R, tvec.reshape(3, 1)], axis=1).astype(np.float32)
    projected, _ = cv2.projectPoints(object_points[inliers.reshape(-1)], rvec, tvec, K_np, None)
    err = np.linalg.norm(projected.reshape(-1, 2) - image_points[inliers.reshape(-1)], axis=1)
    E = torch.from_numpy(E_np).to(device=device, dtype=dtype)
    return E, int(len(inliers)), float(np.median(err))


def estimate_se3_from_flow_depth(
    depth: torch.Tensor,
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    intrinsics: Optional[Union[Tuple[float, float, float, float], torch.Tensor, np.ndarray]] = None,
    stride: int = 8,
    ransac_threshold: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Estimate prev/center/next OpenCV-style extrinsics from MEMFOF flows.

    The centre camera is identity. Prev and next are robust PnP estimates using
    centre-frame 3D points from depth and target 2D correspondences from flow.
    """

    if depth.dim() == 3:
        depth_b = depth.unsqueeze(0)
    else:
        depth_b = depth
    _, _, H_img, W_img = depth_b.shape
    K = make_intrinsics(H_img, W_img, intrinsics, device=depth_b.device, dtype=depth_b.dtype)
    E_center = torch.eye(3, 4, device=depth_b.device, dtype=depth_b.dtype)
    E_next, inliers_next, err_next = _estimate_relative_pose_pnp(
        flow_fwd,
        depth_b,
        K,
        stride=stride,
        ransac_threshold=ransac_threshold,
    )
    E_prev, inliers_prev, err_prev = _estimate_relative_pose_pnp(
        flow_bwd,
        depth_b,
        K,
        stride=stride,
        ransac_threshold=ransac_threshold,
    )
    extrinsics = torch.stack([E_prev, E_center, E_next], dim=0)
    diagnostics = {
        "se3_estimator": "pnp_ransac_from_flow_depth",
        "se3_stride": stride,
        "se3_ransac_threshold": ransac_threshold,
        "se3_inliers_prev": inliers_prev,
        "se3_inliers_next": inliers_next,
        "se3_median_reproj_prev": err_prev,
        "se3_median_reproj_next": err_next,
    }
    return extrinsics, K, diagnostics


def pose_to_motion_field(
    extrinsics: Union[torch.Tensor, np.ndarray],  # [3, 3, 4] prev, center, next (OpenCV w2c)
    intrinsics: Union[torch.Tensor, np.ndarray],  # [3, 3] K for center
    depth: torch.Tensor,  # [B, 1, H, W] or [1, H, W] center-view depth (metric)
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute camera flow from DA3 pose + depth (SE(3) projection).

    Center pixel (u,v) with depth d -> unproject to 3D -> world -> project to prev/next.
    Returns cam_fwd (center->next), cam_bwd (center->prev) as displacement [B, 2, H, W].

    Args:
        extrinsics: [3, 3, 4] E_prev, E_center, E_next (OpenCV w2c)
        intrinsics: [3, 3] K
        depth: [B, 1, H, W] center depth
        eps: min depth to avoid div-by-zero

    Returns:
        cam_fwd: [B, 2, H, W] center -> next displacement
        cam_bwd: [B, 2, H, W] center -> prev displacement
    """
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.from_numpy(extrinsics).float()
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics).float()

    device = depth.device
    dtype = depth.dtype
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)

    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(0) if depth.shape[0] == 1 else depth.unsqueeze(1)
    elif depth.dim() == 4 and depth.shape[1] != 1:
        depth = depth[:, :1]

    B, _, H, W = depth.shape
    depth_safe = depth.clamp(min=eps)

    E_prev = extrinsics[0]   # [3, 4]
    E_center = extrinsics[1]
    E_next = extrinsics[2]
    K = intrinsics

    # Pixel grid [H, W]
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    ones = torch.ones_like(xx)
    # [3, H*W]
    uv1 = torch.stack([xx.flatten(), yy.flatten(), ones.flatten()], dim=0)

    # K^{-1} @ [u,v,1]^T
    K_inv = torch.linalg.inv(K)
    rays = (K_inv @ uv1).unsqueeze(0).expand(B, -1, -1)  # [B, 3, H*W]

    # P_c = depth * K^{-1} @ [u,v,1]
    depth_flat = depth_safe.view(B, 1, -1)  # [B, 1, H*W]
    P_c = rays * depth_flat  # [B, 3, H*W]

    # World: P_w = R_center^T (P_c - t_center). Batch-friendly.
    R_c = E_center[:3, :3].T.unsqueeze(0).expand(B, 3, 3)  # [B, 3, 3]
    t_c = E_center[:3, 3].view(1, 3, 1).expand(B, 3, -1)
    P_c_ = P_c - t_c  # [B, 3, H*W]
    P_w = torch.bmm(R_c, P_c_)  # [B, 3, H*W]

    # Prev: P_prev = R_prev @ P_w + t_prev; project with K
    R_p = E_prev[:3, :3].unsqueeze(0).expand(B, 3, 3)
    t_p = E_prev[:3, 3].view(1, 3, 1).expand(B, 3, 1)
    P_prev = torch.bmm(R_p, P_w) + t_p  # [B, 3, H*W]
    K_b = K.unsqueeze(0).expand(B, 3, 3)
    p_prev = torch.bmm(K_b, P_prev)  # [B, 3, H*W]
    z_p = p_prev[:, 2:3].clamp(min=eps)
    u_prev = (p_prev[:, 0:1] / z_p).view(B, 1, H, W)
    v_prev = (p_prev[:, 1:2] / z_p).view(B, 1, H, W)

    # Next: same for E_next
    R_n = E_next[:3, :3].unsqueeze(0).expand(B, 3, 3)
    t_n = E_next[:3, 3].view(1, 3, 1).expand(B, 3, 1)
    P_next = torch.bmm(R_n, P_w) + t_n
    p_next = torch.bmm(K_b, P_next)
    z_n = p_next[:, 2:3].clamp(min=eps)
    u_next = (p_next[:, 0:1] / z_n).view(B, 1, H, W)
    v_next = (p_next[:, 1:2] / z_n).view(B, 1, H, W)

    xx_ = xx.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W).to(device=device, dtype=dtype)
    yy_ = yy.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W).to(device=device, dtype=dtype)

    cam_bwd = torch.cat([u_prev - xx_, v_prev - yy_], dim=1)  # [B, 2, H, W] center->prev
    cam_fwd = torch.cat([u_next - xx_, v_next - yy_], dim=1)  # [B, 2, H, W] center->next

    return cam_fwd, cam_bwd


def extract_object_masks_from_residual(
    residual_flow: torch.Tensor,  # [B, 2, H, W] or [B, H, W, 2]
    threshold: float = 2.0,  # pixels
    min_area: int = 100,  # minimum pixels per mask
    max_objects: int = 5,  # maximum number of objects to extract
) -> torch.Tensor:
    """Extract object masks from residual flow (flow - camera_motion).
    
    Objects are regions where residual flow magnitude exceeds threshold.
    Uses connected components to find separate objects.
    
    Args:
        residual_flow: [B, 2, H, W] or [B, H, W, 2] residual flow
        threshold: magnitude threshold in pixels
        min_area: minimum area for a valid mask
        max_objects: maximum number of objects per image
        
    Returns:
        masks: [B, K, H, W] binary masks (K = max_objects, padded with zeros)
    """
    try:
        from scipy.ndimage import label
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if residual_flow.dim() == 4 and residual_flow.shape[-1] == 2:
        # [B, H, W, 2] -> [B, 2, H, W]
        residual_flow = residual_flow.permute(0, 3, 1, 2)
    
    B, _, H, W = residual_flow.shape
    device = residual_flow.device
    dtype = residual_flow.dtype
    
    # Compute magnitude
    mag = torch.norm(residual_flow, dim=1)  # [B, H, W]
    
    # Threshold
    object_mask = (mag > threshold).float()  # [B, H, W]
    
    all_masks = []
    for b in range(B):
        obj_b = object_mask[b].cpu().numpy()  # [H, W]
        
        if obj_b.sum() < min_area:
            # No objects
            all_masks.append(torch.zeros(max_objects, H, W, device=device, dtype=dtype))
            continue
        
        if has_scipy:
            # Connected components
            labeled, num_features = label(obj_b > 0.5)
            masks_b = []
            for i in range(1, num_features + 1):
                mask_i = (labeled == i).astype(float)
                if mask_i.sum() >= min_area:
                    masks_b.append(torch.from_numpy(mask_i).to(device).to(dtype))
                if len(masks_b) >= max_objects:
                    break
        else:
            # Fallback: single mask
            masks_b = [torch.from_numpy(obj_b).to(device).to(dtype)]
        
        # Pad to max_objects
        while len(masks_b) < max_objects:
            masks_b.append(torch.zeros(H, W, device=device, dtype=dtype))
        
        masks_b = masks_b[:max_objects]
        all_masks.append(torch.stack(masks_b, dim=0))  # [K, H, W]
    
    masks = torch.stack(all_masks, dim=0)  # [B, K, H, W]
    return masks
