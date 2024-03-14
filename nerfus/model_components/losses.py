"""
Collection of Losses.
"""
from enum import Enum
from typing import Dict, Literal, Optional, Tuple, Callable, cast
from math import log, exp

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.math import masked_reduction, normalized_depth_scale_and_shift


def uncertainty_loss(gt_image, pred_image, betas, weights_list):
    """From NeRFUS"""
    weights = weights_list[-1][..., 0]
    loss = 0.5 * torch.sum((gt_image - pred_image) ** 2 / betas ** 2, dim=-1)
    loss += 0.5 * torch.log(betas[..., 0] ** 2)
    loss += 0.01 * weights.mean(dim=-1)

    return torch.mean(loss)


class EDLLoss(nn.Module):
    """Loss for semantic segmentation using evidential learning"""
    def __init__(
        self,
        num_classes: int,
        evidence_function: Callable[[Tensor], Tensor],
        loss_type: Literal["max_likelihood", "mse_bayes_risk", "cross_entropy_bayes_risk", "cross_entropy"],
        with_kl_div: bool = True,
        with_avu_loss: bool = True,
        disentangle: bool = False,
        annealing_method: Literal["step", "exp"] = "step",
        init_annealing_factor: float = 0.001,
        annealing_step: int = 10000,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.evidence_function = evidence_function
        self.loss_type = loss_type
        self.with_kl_div = with_kl_div
        self.with_avu_loss = with_avu_loss
        self.disentangle = disentangle

        # annealing
        self.annealing_method = annealing_method
        self.cur_iter_num = 0
        self.init_annealing_factor = init_annealing_factor
        self.annealing_step = annealing_step
        self.eps = 1e-10

    def forward(
        self,
        prediction: Tensor,
        target: Tensor
    ) -> Float[Tensor, "0"]:
        evidence = self.evidence_function(prediction)
        alpha = evidence + 1

        one_hot_target = torch.eye(self.num_classes).to(prediction)[target]

        if self.loss_type == "max_likelihood":
            loss = self._max_likelihood(alpha, one_hot_target)
        elif self.loss_type == "mse_bayes_risk":
            loss = self._mse_bayes_risk(alpha, one_hot_target)
        elif self.loss_type == "cross_entropy_bayes_risk":
            loss = self._cross_entropy_bayes_risk(alpha, one_hot_target)
        elif self.loss_type == "cross_entropy":
            loss = self._cross_entropy(alpha, target)
        else:
            raise NotImplementedError

        annealing_coef = self._compute_annealing_coef()

        if self.with_kl_div:
            loss += annealing_coef * self._kl_divergence(alpha, one_hot_target)

        if self.with_avu_loss:
            loss += self._avu_loss(alpha, target, annealing_coef)

        self.cur_iter_num += 1

        return loss.mean()

    @staticmethod
    def _max_likelihood(alpha, target):
        strength = alpha.sum(dim=-1, keepdim=True)
        loss = (target * (strength.log() - alpha.log())).sum(dim=-1)

        return loss

    @staticmethod
    def _mse_bayes_risk(alpha, target):
        strength = alpha.sum(dim=-1, keepdim=True)
        pred_prob = alpha / strength

        err = (target - pred_prob) ** 2
        var = pred_prob * (1 - pred_prob) / (strength + 1)

        loss = (err + var).sum(dim=-1)

        return loss

    @staticmethod
    def _cross_entropy_bayes_risk(alpha, target):
        strength = alpha.sum(dim=-1, keepdim=True)
        loss = (target * (strength.digamma() - alpha.digamma())).sum(dim=-1)

        return loss

    @staticmethod
    def _cross_entropy(alpha, target):
        strength = alpha.sum(dim=-1, keepdim=True)
        pred_prob = alpha / strength
        log_pred_prob = alpha.log() - strength.log()
        err = nn.functional.nll_loss(log_pred_prob, target, reduction="none")
        var = torch.sum(pred_prob * (1 - pred_prob) / (strength + 1), dim=-1, keepdim=True)

        loss = err + var

        return loss

    def _compute_annealing_coef(self):
        if self.annealing_method == 'step':
            annealing_coef = torch.min(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(self.cur_iter_num / self.annealing_step, dtype=torch.float32)
            )
        elif self.annealing_method == 'exp':
            annealing_start = torch.tensor(self.init_annealing_factor, dtype=torch.float32)
            annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / 60000 * self.cur_iter_num)
        else:
            raise NotImplementedError

        return annealing_coef

    def _kl_divergence(self, alpha, target):
        alpha_tilde = target + (1 - target) * alpha
        strength_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

        first = torch.lgamma(alpha_tilde.sum(dim=-1)) \
            - torch.lgamma(alpha_tilde.new_tensor(float(self.num_classes))) \
            - (torch.lgamma(alpha_tilde)).sum(dim=-1)
        second = torch.sum((alpha_tilde - 1) * (alpha_tilde.digamma() - strength_tilde.digamma()), dim=-1)

        loss = first + second

        return loss

    def _avu_loss(self, alpha, target, annealing_coef):
        strength = alpha.sum(dim=-1, keepdim=True)
        pred_scores, pred_cls = torch.max(alpha / strength, dim=-1, keepdim=True)
        uncertainty_mass = self.num_classes / strength

        match = pred_cls.eq(target.unsqueeze(dim=-1)).to(torch.float)

        if self.disentangle:
            acc_uncertain = -torch.log(pred_scores * (1 - uncertainty_mass) + self.eps)
            inacc_certain = -torch.log((1 - pred_scores) * uncertainty_mass + self.eps)
        else:
            acc_uncertain = - pred_scores * torch.log(1 - uncertainty_mass + self.eps)
            inacc_certain = - (1 - pred_scores) * torch.log(uncertainty_mass + self.eps)

        loss = annealing_coef * match * acc_uncertain + (1 - annealing_coef) * (1 - match) * inacc_certain
        return loss.squeeze()
