from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


def load_pose_template(
    pose_dir: str,
    pose_style: str,
    num_frames: int,
    width: int,
    height: int,
) -> torch.Tensor:
    """Load pose template .npy files and build poses_tensor.

    Pose templates are directories of .npy files under pose_dir/pose_style/.
    If the template has fewer frames than needed, it loops.

    Returns tensor of shape (1, 3, num_frames, height, width).
    """
    # Import from echomimic_v2 source
    from src.utils.draw_utils import draw_pose_select_v2

    style_dir = Path(pose_dir) / pose_style
    if not style_dir.exists():
        raise FileNotFoundError(f"Pose style directory not found: {style_dir}")

    # Collect all .npy files sorted
    npy_files = sorted(style_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {style_dir}")

    logger.info("Loading %d pose frames from %s (need %d)", len(npy_files), style_dir, num_frames)

    to_tensor = transforms.ToTensor()
    pose_list = []

    for i in range(num_frames):
        # Loop through available poses if we need more frames
        npy_path = npy_files[i % len(npy_files)]
        pose_data = np.load(str(npy_path), allow_pickle=True).tolist()
        pose_img = draw_pose_select_v2(pose_data, height, width)
        pose_tensor = to_tensor(pose_img)
        pose_list.append(pose_tensor)

    # Stack: list of (3, H, W) -> (3, num_frames, H, W) -> (1, 3, num_frames, H, W)
    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    logger.info("Poses tensor shape: %s", poses_tensor.shape)
    return poses_tensor
