"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfus.nerfus_datamanager import (
    NeRFUSDataManagerConfig,
)
from nerfus.nerfus_model import NeRFUSModelConfig
from nerfus.nerfus_pipeline import (
    NeRFUSPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfus.data.dataparsers.semantic_replica_dataparser import SemanticReplicaDataParserConfig
from nerfus.data.datasets.semantic_segmentation_dataset import SemanticSegmentationDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


nerfus_ngp = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfus",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=NeRFUSPipelineConfig(
            datamanager=NeRFUSDataManagerConfig(
                _target=VanillaDataManager[SemanticSegmentationDataset],
                dataparser=SemanticReplicaDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NeRFUSModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio method template.",
)
