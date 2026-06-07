import os
import random
from glob import glob
from time import sleep

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import retry


def parse_gopro(split="train", data_path="/mnt/ssd2/25e_bog/datasets/GoPro", blur_name="blur_gamma"):
    """
    - sharp/blur: [-1, 1]
    - depth: [-1, ~250] - from DA3 (in meters)
    - flow: UV, vector length <= 1
    """
    split_path = os.path.join(data_path, split) 
    blur_paths, sharp_paths, depth_paths = [], [], []
    flow_fwd_paths, flow_bwd_paths = [], []
    sharp_prev_paths, sharp_next_paths = [], []
    flow_cp_paths, flow_cn_paths = [], []
    
    for video in sorted(os.listdir(split_path)):
        video_dir = os.path.join(split_path, video)

        sharp_dir = os.path.join(video_dir, "sharp")
        blur_dir = os.path.join(video_dir, blur_name)
        depth_dir = os.path.join(video_dir, "depth")
        flow_fwd_dir = os.path.join(video_dir, "flow_forward")
        flow_bwd_dir = os.path.join(video_dir, "flow_backward")
        flow_cp_dir = os.path.join(video_dir, "flow_cp_pix")
        flow_cn_dir = os.path.join(video_dir, "flow_cn_pix")
        
        sharp_frames = sorted(glob(os.path.join(sharp_dir, "*.png")))
        blur_frames = sorted(glob(os.path.join(blur_dir, "*.png")))
        depth_frames = sorted(glob(os.path.join(depth_dir, "*.npy"))) if os.path.exists(depth_dir) else []
        flow_fwd_frames = sorted(glob(os.path.join(flow_fwd_dir, "*.npy"))) if os.path.exists(flow_fwd_dir) else []
        flow_bwd_frames = sorted(glob(os.path.join(flow_bwd_dir, "*.npy"))) if os.path.exists(flow_bwd_dir) else []
        flow_cp_frames = sorted(glob(os.path.join(flow_cp_dir, "*.npy"))) if os.path.exists(flow_cp_dir) else []
        flow_cn_frames = sorted(glob(os.path.join(flow_cn_dir, "*.npy"))) if os.path.exists(flow_cn_dir) else []
                
        # Use frames [1:-1] to have prev/next available
        sharp_paths.extend(sharp_frames[1:-1])
        blur_paths.extend(blur_frames[1:-1])
        sharp_prev_paths.extend(sharp_frames[0:-2])  # prev frame
        sharp_next_paths.extend(sharp_frames[2:])   # next frame
        
        if depth_frames:
            depth_paths.extend(depth_frames[1:-1])
        if flow_fwd_frames:
            flow_fwd_paths.extend(flow_fwd_frames)
        if flow_bwd_frames:
            flow_bwd_paths.extend(flow_bwd_frames)
        if flow_cp_frames:
            flow_cp_paths.extend(flow_cp_frames)
        if flow_cn_frames:
            flow_cn_paths.extend(flow_cn_frames)
    
    return { 
        "blur": blur_paths, 
        "sharp": sharp_paths,
        "sharp_prev": sharp_prev_paths,
        "sharp_next": sharp_next_paths,
        "depth": depth_paths,
        "flow_fwd": flow_fwd_paths,
        "flow_bwd": flow_bwd_paths,
        "flow_cp_pix": flow_cp_paths,
        "flow_cn_pix": flow_cn_paths,
        "img_range": 2.0
    }
    

class BaseDataset(Dataset):
    def __init__(self, dataset_name, split="train", data_path="/mnt/ssd2/25e_bog/datasets/GoPro", transform=None):
        # split = "train" or "test"
        if dataset_name == "gopro":
            parsed = parse_gopro(split, data_path=data_path)
        elif dataset_name == "REDS":
            parsed = parse_gopro(split, data_path=data_path, blur_name="blur")
        else:
            raise ValueError
        
        self.img_range = parsed.get("img_range") # image pixel value range
        self.blur_paths = parsed.get("blur")
        self.sharp_paths = parsed.get("sharp")

        self.sharp_prev_paths = parsed.get("sharp_prev", [])
        self.sharp_next_paths = parsed.get("sharp_next", [])
        self.depth_paths = parsed.get("depth", [])
        
        # flow = UV field (2 channels)
        # scaling:
        # 1) flow /= 147  (10% of 1280x720 diagonal)
        # 2) flow[vector_len > 1] /= vector_len
        self.flow_fwd_paths = parsed.get("flow_fwd", [])
        self.flow_bwd_paths = parsed.get("flow_bwd", [])
        
        # Phase-1 flow: pixel units, cur->prev and cur->next
        self.flow_cp_pix_paths = parsed.get("flow_cp_pix", [])
        self.flow_cn_pix_paths = parsed.get("flow_cn_pix", [])
        
        # Occlusion masks (optional)
        self.occ_cp_paths = parsed.get("occ_cp", [])
        self.occ_cn_paths = parsed.get("occ_cn", [])
        
        self.transform = transform

    def __len__(self):
        return len(self.sharp_paths)

    def __getitem__(self, idx):
        blur = get_img(self.blur_paths[idx])
        sharp = get_img(self.sharp_paths[idx])
        h, w = blur.shape[:2]
        
        # Phase-1 inputs: sharp_prev, sharp_next
        if idx < len(self.sharp_prev_paths):
            sharp_prev = get_img(self.sharp_prev_paths[idx])
        else:
            sharp_prev = sharp.copy()  # Fallback to same frame
        
        if idx < len(self.sharp_next_paths):
            sharp_next = get_img(self.sharp_next_paths[idx])
        else:
            sharp_next = sharp.copy()  # Fallback to same frame
        
        # Phase-1 flows: flow_cp_pix (backward: curr->prev), flow_cn_pix (forward: curr->next)
        # These are stored as [2, H, W] in .npy files (in pixel units)
        if idx < len(self.flow_cp_pix_paths):
            flow_cp_pix = get_npy_flow(self.flow_cp_pix_paths[idx])
        else:
            flow_cp_pix = np.zeros((h, w, 2), dtype=np.float32)
        
        if idx < len(self.flow_cn_pix_paths):
            flow_cn_pix = get_npy_flow(self.flow_cn_pix_paths[idx])
        else:
            flow_cn_pix = np.zeros((h, w, 2), dtype=np.float32)
        
        # Occlusion masks (optional, float32 0/1)
        if idx < len(self.occ_cp_paths):
            occ_cp = get_npy_flow(self.occ_cp_paths[idx])
        else:
            occ_cp = np.ones((h, w, 1), dtype=np.float32)
        
        if idx < len(self.occ_cn_paths):
            occ_cn = get_npy_flow(self.occ_cn_paths[idx])
        else:
            occ_cn = np.ones((h, w, 1), dtype=np.float32)
        
        # Handle optional depth and flow data
        if idx < len(self.depth_paths):
            depth = get_npy(self.depth_paths[idx])
        else:
            depth = np.zeros((h, w, 1), dtype=np.float32)
        
        if idx < len(self.flow_fwd_paths):
            flow_fwd = get_npy(self.flow_fwd_paths[idx])
        else:
            flow_fwd = np.zeros((h, w, 2), dtype=np.float32)
        
        if idx < len(self.flow_bwd_paths):
            flow_bwd = get_npy(self.flow_bwd_paths[idx])
        else:
            flow_bwd = np.zeros((h, w, 2), dtype=np.float32)
        
        sample = {
            "blur": blur, 
            "sharp": sharp,
            "sharp_prev": sharp_prev,
            "sharp_next": sharp_next,
            "flow_cp": flow_cp_pix,  # [H, W, 2] in pixels
            "flow_cn": flow_cn_pix,  # [H, W, 2] in pixels
            "occ_cp": occ_cp,        # [H, W, 1] binary mask
            "occ_cn": occ_cn,        # [H, W, 1] binary mask
            "depth": depth,
            "flow_fwd": flow_fwd,
            "flow_bwd": flow_bwd
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def rotation_matrix(angle):
        rad = np.radians(angle)
        
        cos_theta = np.cos(rad)
        sin_theta = np.sin(rad)
        rot_matrix = np.array([[cos_theta, -sin_theta],
                               [sin_theta, cos_theta]])
        
        return rot_matrix


class RandomRotate(object):
    """Random 90-degree rotation. Rotates flow vectors accordingly."""
    FLOW_KEYS = {'flow_fwd', 'flow_bwd', 'flow_cp', 'flow_cn'}
    
    def __call__(self, data):
        dirct = random.randint(0, 3)
        
        for key in data.keys():
            data[key] = np.rot90(data[key], dirct).copy()
            
            if key in self.FLOW_KEYS:
                # Rotate the flow vectors themselves
                vectors = data[key].copy()
                orig_shape = vectors.shape
                vectors = vectors.reshape((-1, 2))
                rot_matrix = rotation_matrix(-90 * dirct)
                rotated_vectors = (rot_matrix @ vectors.T).T
                data[key] = rotated_vectors.reshape(orig_shape)
                
        return data

    
class RandomFlip(object):
    """Random horizontal/vertical flip. Flips flow vectors accordingly."""
    FLOW_KEYS = {'flow_fwd', 'flow_bwd', 'flow_cp', 'flow_cn'}
    
    def __call__(self, data):
        # Horizontal flip
        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.fliplr(data[key]).copy()
                if key in self.FLOW_KEYS:
                    data[key][:, :, 0] = -data[key][:, :, 0]  # Flip x component
        
        # Vertical flip
        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.flipud(data[key]).copy()
                if key in self.FLOW_KEYS:
                    data[key][:, :, 1] = -data[key][:, :, 1]  # Flip y component
        
        return data


class RandomCrop(object):
    def __init__(self, Hsize, Wsize):
        self.size = (Hsize, Wsize)

    def __call__(self, data):
        H, W, _ = np.shape(list(data.values())[0])
        h, w = self.size
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        for key in data.keys():
            data[key] = data[key][top : top + h, left : left + w].copy()
        return data


class Normalize(object):
    """Normalize images to [-1, 1] and transpose to [C, H, W]."""
    IMAGE_KEYS = {'sharp', 'blur', 'sharp_prev', 'sharp_next'}
    FLOW_KEYS = {'flow_fwd', 'flow_bwd', 'flow_cp', 'flow_cn'}
    MASK_KEYS = {'occ_cp', 'occ_cn', 'depth'}
    
    def __call__(self, data):
        # Normalize and transpose images [H, W, C] -> [C, H, W]
        for key in self.IMAGE_KEYS:
            if key in data:
                data[key] = ((data[key] / 255.0) * 2 - 1.0).astype(np.float32)
                data[key] = data[key].transpose(2, 0, 1)
        
        # Transpose flow fields [H, W, 2] -> [2, H, W] (keep in pixel units)
        for key in self.FLOW_KEYS:
            if key in data:
                data[key] = data[key].astype(np.float32).transpose(2, 0, 1)
        
        # Transpose masks [H, W, C] -> [C, H, W]
        for key in self.MASK_KEYS:
            if key in data:
                data[key] = data[key].astype(np.float32).transpose(2, 0, 1)
            
        return data  # out: (C, H, W) for each tensor


def get_default_transforms(patch_size):
    train_transform = transforms.Compose([
        RandomCrop(patch_size, patch_size),
        RandomFlip(),
        RandomRotate(),
        Normalize(),
    ])
    val_transform = transforms.Compose([Normalize()])
    
    return train_transform, val_transform


@retry
def get_img(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32)


@retry
def get_npy(path):
    """Load .npy file and ensure it's in [H, W, C] format."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        # Single channel (e.g., depth map): [H, W] -> [H, W, 1]
        arr = arr[..., np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[0] in [1, 2, 3, 4]:  # Likely [C, H, W] format
            arr = arr.transpose(1, 2, 0)  # -> [H, W, C]
        # else already [H, W, C]
    return arr


@retry
def get_npy_flow(path):
    """Load flow .npy file. Handles both [2, H, W] and [H, W, 2] formats."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 2:  # [2, H, W] -> [H, W, 2]
            arr = arr.transpose(1, 2, 0)
        elif arr.shape[-1] == 2:  # already [H, W, 2]
            pass
        elif arr.shape[-1] == 1:  # occlusion mask [H, W, 1]
            pass
        elif arr.shape[0] == 1:  # occlusion mask [1, H, W] -> [H, W, 1]
            arr = arr.transpose(1, 2, 0)
    elif arr.ndim == 2:  # [H, W] -> [H, W, 1]
        arr = arr[..., None]
    return arr
