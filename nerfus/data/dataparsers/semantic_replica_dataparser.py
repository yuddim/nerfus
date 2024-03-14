# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Datapaser for semantic replica formatted data"""

from __future__ import annotations

from typing import Dict, Tuple, Type, Literal

import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
from imgviz import label_colormap

import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
    get_train_eval_split_all,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SemanticReplicaDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: SemanticReplica)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    eval_mode: Literal["fraction", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    include_semantics: bool = True
    """whether or not to include loading of semantics data"""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    sparse_data_factor: float = 1.0
    """Selects the specified percentage of scene frames"""


@dataclass
class SemanticReplica(DataParser):
    """SemanticReplica DatasetParser"""

    config: SemanticReplicaDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        traj_file = self.config.data / "traj_w_c.txt"
        image_dir = self.config.data / "rgb"
        depth_dir = self.config.data / "depth"
        semantic_class_dir = self.config.data / "semantic_class"

        image_filenames = list(sorted(image_dir.iterdir(), key=lambda file_name: int(file_name.name.split("_")[-1][:-4])))
        depth_filenames = list(sorted(depth_dir.iterdir(), key=lambda file_name: int(file_name.name.split("_")[-1][:-4])))
        poses = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        # find train and eval indices based on the eval_mode specified
        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
        elif self.config.eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        # find camera parameters
        first_img = cv2.imread(str(image_filenames[0].absolute()))  # type: ignore
        h, w, _ = first_img.shape

        hfov = 90
        # the pin-hole camera has the same value for fx and fy
        fx = w / 2.0 / np.tan(np.radians(hfov / 2.0))
        fy = fx
        cx = (w - 1.0) / 2.0
        cy = (h - 1.0) / 2.0

        poses[:, :3, 1] *= -1
        poses[:, :3, 2] *= -1
        poses = torch.from_numpy(poses).float()

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method="up",
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor
        poses = poses[indices.tolist()]

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=h,
            width=w,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # semantic
        semantics = None

        if self.config.include_semantics:
            semantic_class_filenames = list(sorted(semantic_class_dir.iterdir(), key=lambda file_name: int(file_name.name.split("_")[-1][:-4])))
            semantic_classes = load_from_json(self.config.data / "classes.json")["classes"]
            colors = torch.tensor(label_colormap()[:len(semantic_classes)]) / 255.
            semantic_class_filenames = [semantic_class_filenames[i] for i in indices]

            semantics = Semantics(filenames=semantic_class_filenames, classes=semantic_classes, colors=colors)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            metadata={"semantics": semantics} if self.config.include_semantics else {},
        )

        return dataparser_outputs
