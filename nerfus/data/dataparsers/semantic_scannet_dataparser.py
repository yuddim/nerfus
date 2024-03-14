"""Data parser for ScanNet dataset"""
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import cv2
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class SparseScanNetDataParserConfig(DataParserConfig):
    """Sparse ScanNet dataset config.
    ScanNet dataset (https://www.scan-net.org/) is a large-scale 3D dataset of indoor scenes.
    This dataparser assumes that the dense stream was extracted from .sens files.
    Expected structure of scene directory:

        root/
        ├── color/
        ├── depth/
        ├── intrinsic/
        ├── pose/
        ├── semantics/
    """

    _target: Type = field(default_factory=lambda: SparseScanNet)
    """target class to instantiate"""
    data: Path = Path("data/scannet/scene0423_02")
    """Path to ScanNet folder with densely extracted scenes."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    sparse_data_factor: float = 0.1
    """Selects the specified percentage of scene frames"""


@dataclass
class SparseScanNet(DataParser):
    """ScanNet DatasetParser"""

    config: SparseScanNetDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        image_dir = self.config.data / "color"
        depth_dir = self.config.data / "depth"
        pose_dir = self.config.data / "pose"
        semantics_dir = self.config.data / "semantics"

        img_dir_sorted = list(sorted(image_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))
        depth_dir_sorted = list(sorted(depth_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))
        pose_dir_sorted = list(sorted(pose_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))
        semantics_dir_sorted = list(sorted(semantics_dir.iterdir(),  key=lambda x: int(x.name.split(".")[0])))

        count_use_images = int(self.config.sparse_data_factor * len(img_dir_sorted))
        use_images_index = np.linspace(0, len(img_dir_sorted) - 1, count_use_images, dtype=int)

        img_dir_sorted = [img_dir_sorted[idx] for idx in use_images_index]
        depth_dir_sorted = [depth_dir_sorted[idx] for idx in use_images_index]
        pose_dir_sorted = [pose_dir_sorted[idx] for idx in use_images_index]
        semantics_dir_sorted = [semantics_dir_sorted[idx] for idx in use_images_index]

        first_img = cv2.imread(str(img_dir_sorted[0].absolute()))  # type: ignore
        h, w, _ = first_img.shape

        image_filenames, depth_filenames, semantics_filenames, intrinsics, poses = [], [], [], [], []

        K = np.loadtxt(self.config.data / "intrinsic" / "intrinsic_color.txt")
        for img, depth, pose, semantics in zip(img_dir_sorted, depth_dir_sorted, pose_dir_sorted, semantics_dir_sorted):
            pose = np.loadtxt(pose).reshape(4, 4)
            pose[:3, 1] *= -1
            pose[:3, 2] *= -1
            pose = torch.from_numpy(pose).float()
            # We cannot accept files directly, as some of the poses are invalid
            if np.isinf(pose).any():
                continue

            intrinsic = K.copy()
            intrinsic[1, 2] += 2

            poses.append(pose)
            intrinsics.append(intrinsic)
            image_filenames.append(img)
            depth_filenames.append(depth)
            semantics_filenames.append(semantics)

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_fraction)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.stack(poses).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method="none",
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        # if self.config.auto_scale_poses:
        #     scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        # scale_factor *= self.config.scale_factor
        #
        # poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        semantics_filenames = [semantics_filenames[i] for i in indices]
        intrinsics = intrinsics[indices.tolist()]
        poses = poses[indices.tolist()]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox.from_camera_poses(poses, aabb_scale)

        # --- semantics ---
        classes_data = load_from_json(self.config.data / "classes.json")
        classes = classes_data["classes"]
        colors = torch.tensor(classes_data["color_classes"], dtype=torch.float32) / 255.0
        semantics = Semantics(filenames=semantics_filenames, classes=classes, colors=colors)

        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=h,
            width=w,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "semantics": semantics
            }
        )

        return dataparser_outputs
