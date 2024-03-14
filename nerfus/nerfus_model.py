"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple, List, Union, Literal

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure, confusion_matrix
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.classification import MulticlassCalibrationError
from torch_uncertainty.metrics import AUSE

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfus.nerfus_field import NeRFUSField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    scale_gradients_by_distance_squared,
)
from nerfus.model_components.losses import EDLLoss, uncertainty_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler, VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    UncertaintyRenderer,
    RGBRenderer,
    SemanticRenderer
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.base_model import Model
from nerfstudio.utils import colormaps


@dataclass
class NeRFUSModelConfig(NerfactoModelConfig):
    """NeRFUS Model Configuration"""

    _target: Type = field(default_factory=lambda: NeRFUSModel)
    semantics_loss_mult: float = 1.0
    """Semantics loss multiplier."""
    evidential_loss_type: Literal["max_likelihood", "mse_bayes_risk", "cross_entropy_bayes_risk", "cross_entropy"] = "max_likelihood"
    """Loss type for evidential learning"""
    annealing_method: Literal["step", "exp"] = "exp"
    """The type of smoothing of the regularizing term in the loss for for evidential learning"""
    init_annealing_factor: float = 0.01
    """The initial value of the annealing factor for exponential smoothing. Has a value between 0 and 1"""
    evidence_function: Literal["relu", "softmax", "softplus"] = "softplus"
    """evidence_function"""
    annealing_step: int = 20000
    """Number of steps for linear smoothing"""
    pass_semantic_gradients: bool = False
    """pass_semantic_gradients"""
    use_semantics: bool = True
    """Whether to use semantics."""
    use_color_uncertainty: bool = True
    """Whether to use color uncertainty."""
    use_aleatoric_semantics_uncertainty: bool = True
    """Whether to use aleatoric semantics uncertainty."""
    use_epistemic_semantics_uncertainty: bool = True
    """Whether to use aleatoric semantics uncertainty."""
    hidden_dim_semantic: int = 64
    """Dimension of hidden layers for semantic network"""
    min_uncertainty: float = 0.01
    """Minimum value for uncertainty"""
    color_uncertainty_loss_mult: float = 0.5
    """Color uncertainty loss multiplier."""


class NeRFUSModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NeRFUSModelConfig

    def __init__(self, config: NeRFUSModelConfig, metadata: Dict, **kwargs) -> None:
        use_semantics = config.use_semantics

        if use_semantics:
            assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
            self.semantics = metadata["semantics"]
            self.num_semantic_classes = len(self.semantics.classes)
        else:
            self.num_semantic_classes = None

        super().__init__(config=config, **kwargs)

        if use_semantics:
            self.colormap = self.semantics.colors.clone().detach().to(self.device)

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = NeRFUSField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            use_semantics=self.config.use_semantics,
            use_color_uncertainty=self.config.use_color_uncertainty,
            use_semantics_uncertainty=self.config.use_aleatoric_semantics_uncertainty,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_semantic=self.config.hidden_dim_semantic,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            num_semantic_classes=self.num_semantic_classes,
            pass_semantic_gradients=self.config.pass_semantic_gradients,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")

        if self.config.use_color_uncertainty:
            self.renderer_color_uncertainty = UncertaintyRenderer()

        if self.config.use_semantics:
            self.renderer_semantics = SemanticRenderer()

            if not self.config.use_epistemic_semantics_uncertainty and self.config.use_aleatoric_semantics_uncertainty:
                self.renderer_aleatoric_semantics_uncertainty = UncertaintyRenderer()

        # evidence
        self.evidence_function = F.softplus

        # losses
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.edl_mse_loss = EDLLoss(
            self.num_semantic_classes,
            self.evidence_function,
            loss_type=self.config.evidential_loss_type,
            annealing_method=self.config.annealing_method,
            init_annealing_factor=self.config.init_annealing_factor,
            annealing_step=self.config.annealing_step,
        )

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.conf_matrix = confusion_matrix
        self.nll_uncertainty = torch.nn.NLLLoss(reduction="mean")
        self.mece = MulticlassCalibrationError(self.num_semantic_classes, norm='l1')
        self.ause = AUSE()
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples)

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # color uncertainty
        if self.config.use_color_uncertainty:
            color_uncertainty = self.renderer_color_uncertainty(field_outputs["color_uncertainty"], weights=weights)
            outputs["color_uncertainty"] = color_uncertainty + self.config.min_uncertainty

        # semantics
        if self.config.use_semantics:
            semantic_weights = weights
            if not self.config.pass_semantic_gradients:
                semantic_weights = weights.detach()

            outputs["semantics"] = self.renderer_semantics(field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights)

            if not self.config.use_epistemic_semantics_uncertainty and self.config.use_aleatoric_semantics_uncertainty:
                betas = self.renderer_aleatoric_semantics_uncertainty(
                    field_outputs["semantic_uncertanty"], weights=semantic_weights
                ) + self.config.min_uncertainty
                outputs["semantics"] += betas * torch.randn(outputs["semantics"].size()).to(outputs["semantics"])
                outputs["aleatoric_semantics_uncertainty"] = -torch.sum(
                    F.softmax(outputs["semantics"], dim=-1) * torch.log_softmax(outputs["semantics"], dim=-1),
                    dim=-1, keepdim=True
                )

            if self.config.use_epistemic_semantics_uncertainty:
                evidence = self.evidence_function(outputs["semantics"])
                alpha = evidence + 1
                strength = torch.sum(alpha, dim=-1, keepdim=True)
                outputs["epistemic_semantics_uncertainty"] = self.num_semantic_classes / strength

                prob = alpha / strength
                log_prob = alpha.log() - strength.log()

                if self.config.use_aleatoric_semantics_uncertainty:
                    outputs["aleatoric_semantics_uncertainty"] = -torch.sum(
                        prob * log_prob, dim=-1, keepdim=True
                    ) / np.log(self.num_semantic_classes)
            else:
                prob = F.softmax(outputs["semantics"], dim=-1)

            semantic_labels = torch.argmax(prob, dim=-1)

            # semantics colormaps
            outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        semantics = batch["semantics"].to(self.device)

        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        if self.config.use_epistemic_semantics_uncertainty:
            evidence = self.evidence_function(outputs["semantics"])
            alpha = evidence + 1
            strength = torch.sum(alpha, dim=-1, keepdim=True)
            prob = alpha / strength
            log_prob = alpha.log() - strength.log()
        else:
            prob = F.softmax(outputs["semantics"], dim=-1)
            log_prob = F.log_softmax(outputs["semantics"], dim=-1)

        metrics_dict["accuracy"] = torch.mean((prob.argmax(dim=-1, keepdim=True) == semantics).to(torch.float))

        if self.config.use_aleatoric_semantics_uncertainty or self.config.use_epistemic_semantics_uncertainty:
            metrics_dict["nll_uncertainty"] = self.nll_uncertainty(log_prob, semantics.squeeze())
            # metrics_dict["mece"] = self.mece(prob, semantics.squeeze())

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]

        # semantic loss
        if self.config.use_semantics:
            if self.config.use_epistemic_semantics_uncertainty:
                loss_dict["semantics_loss"] = self.config.semantics_loss_mult * self.edl_mse_loss(
                    outputs["semantics"], batch["semantics"][..., 0].long().to(self.device)
                )
            else:
                loss_dict["semantics_loss"] = self.config.semantics_loss_mult * self.cross_entropy_loss(
                    outputs["semantics"], batch["semantics"][..., 0].long().to(self.device)
                )

        # uncertainty loss
        if self.training and self.config.use_color_uncertainty:
            loss_dict["uncertainty_loss"] = self.config.color_uncertainty_loss_mult * uncertainty_loss(
                pred_rgb, gt_rgb, outputs["color_uncertainty"], outputs["weights_list"]
            )

        for key, val in loss_dict.items():
            if torch.isnan(val):
                print(key)
                import sys
                sys.exit(1)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, Union[float, torch.Tensor]], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        gt_semantic = batch["semantics"].squeeze().to(self.device)

        if self.config.use_epistemic_semantics_uncertainty:
            evidence = self.evidence_function(outputs["semantics"])
            alpha = evidence + 1
            strength = torch.sum(alpha, dim=-1, keepdim=True)
            prob = alpha / strength
            log_prob = alpha.log() - strength.log()
        else:
            prob = F.softmax(outputs["semantics"], dim=-1)
            log_prob = F.log_softmax(outputs["semantics"], dim=-1)

        predicted_semantics = torch.argmax(prob, dim=-1)
        gt_semantics_colormap = self.colormap.to(self.device)[gt_semantic]
        predicted_semantics_colormap = outputs["semantics_colormap"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_semantics = torch.cat([gt_semantics_colormap, predicted_semantics_colormap], dim=1)


        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        confmat = self.conf_matrix(
            predicted_semantics, gt_semantic, "multiclass",
            num_classes=self.num_semantic_classes
        )

        wmiou = torch.nanmean(confmat.diag() / (confmat.sum(dim=1) + confmat.sum(dim=0) - confmat.diag()))

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips), "confmat": confmat, "wmiou": float(wmiou)}  # type: ignore

        if self.config.use_aleatoric_semantics_uncertainty or self.config.use_epistemic_semantics_uncertainty:
            nll_uncertainty = self.nll_uncertainty(log_prob.permute(2, 0, 1).unsqueeze(0), gt_semantic.unsqueeze(0))
            mece = self.mece(prob.view(1, self.num_semantic_classes, -1), gt_semantic.view(1, -1))
            metrics_dict["nll_uncertainty"] = float(nll_uncertainty.item())
            metrics_dict["mece"] = float(mece.item())

        images_dict = {"img": combined_rgb, "sem": combined_semantics, "accumulation": combined_acc, "depth": combined_depth}

        if self.config.use_color_uncertainty:
            color_uncertainty = colormaps.apply_colormap(outputs["color_uncertainty"])
            combined_color_unc = torch.cat([color_uncertainty], dim=1)
            images_dict["color_uncertainty"] = combined_color_unc

        if self.config.use_epistemic_semantics_uncertainty:
            epistemic_uncertainty = colormaps.apply_colormap(outputs["epistemic_semantics_uncertainty"])
            combined_epistemic_unc = torch.cat([epistemic_uncertainty], dim=1)
            images_dict["epistemic_semantics_uncertainty"] = combined_epistemic_unc
            metrics_dict["ause"] = self.ause(outputs["epistemic_semantics_uncertainty"].view(-1), predicted_semantics.eq(gt_semantic).view(-1).long())

            import matplotlib.pyplot as plt
            import uuid
            fig, ax = plt.subplots(1,1)
            self.ause.plot(ax=ax)
            fig.savefig(f'curves_corresponding/{uuid.uuid4()}.pdf')
            plt.close(fig)


        if self.config.use_aleatoric_semantics_uncertainty:
            aleatoric_uncertainty = colormaps.apply_colormap(outputs["aleatoric_semantics_uncertainty"])
            combined_aleatoric_unc = torch.cat([aleatoric_uncertainty], dim=1)
            images_dict["aleatoric_semantics_uncertainty"] = combined_aleatoric_unc

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict