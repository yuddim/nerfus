"""
NeRFUS Field
"""

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    RGBFieldHead,
    SemanticFieldHead,

    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions


class NeRFUSField(Field):
    """NeRFUS Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
            self,
            aabb: Tensor,
            num_images: int,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_levels: int = 16,
            base_res: int = 16,
            max_res: int = 2048,
            log2_hashmap_size: int = 19,
            num_layers_color: int = 3,
            features_per_level: int = 2,
            hidden_dim_color: int = 64,
            use_semantics: bool = False,
            num_semantic_classes: Optional[int] = None,
            num_layers_semantic: int = 2,
            hidden_dim_semantic: int = 64,
            pass_semantic_gradients: bool = False,
            use_color_uncertainty: bool = False,
            use_semantics_uncertainty: bool = False,
            min_uncertainty: float = 0.01,
            spatial_distortion: Optional[SpatialDistortion] = None,
            implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.use_semantics = use_semantics
        self.use_color_uncertainty = use_color_uncertainty
        self.use_semantics_uncertainty = use_semantics_uncertainty
        self.min_uncertainty = min_uncertainty
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation
        )

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)

        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim + self.direction_encoding.get_out_dim(),
                num_layers=num_layers_semantic,
                layer_width=hidden_dim_semantic,
                out_dim=hidden_dim_semantic,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(), num_classes=num_semantic_classes
            )

            if self.use_semantics_uncertainty:
                self.field_head_semantic_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_semantics.get_out_dim())

        self.mlp_rgb = MLP(
            in_dim=self.geo_feat_dim,
            num_layers=num_layers_color - 1,
            layer_width=hidden_dim_color,
            out_dim=hidden_dim_color,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        if self.use_color_uncertainty:
            self.field_head_color_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_rgb.get_out_dim())

        self.field_head_rgb = RGBFieldHead(in_dim=self.mlp_rgb.get_out_dim())

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
            self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # semantics
        if self.use_semantics:
            semantics_input = torch.cat([density_embedding.view(-1, self.geo_feat_dim), d], dim=-1, )
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)
            if self.use_semantics_uncertainty:
                outputs["semantic_uncertainty"] = self.field_head_semantic_uncertainty(x) + self.min_uncertainty

        # color
        h = self.mlp_rgb(density_embedding.view(-1, self.geo_feat_dim)).view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.RGB] = self.field_head_rgb(h)
        if self.use_color_uncertainty:
            outputs["color_uncertainty"] = self.field_head_color_uncertainty(h) + self.min_uncertainty

        return outputs
