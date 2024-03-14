"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def get_semantics_tensors_from_path(
    filepath: Path, scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    """
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    return semantics
