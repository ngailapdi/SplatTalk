from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch.nn.functional as F 
import torch.nn as nn


@dataclass
class LossKLCfg:
    weight: float


@dataclass
class LossKLCfgWrapper:
    kl: LossKLCfg


class LossKL(Loss[LossKLCfg, LossKLCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        encoder_results: dict,
        global_step: int,
        dr_prediction: DecoderOutput | None = None,
    ) -> Float[Tensor, ""]:
        gt = batch["target"]["feature"]
        pred = prediction.feature
        feat = pred.shape[2]
        pred = pred.permute(0,1,3,4,2).reshape(-1, feat)
        gt = gt.permute(0,1,3,4,2).reshape(-1, feat)
        # print(gt.shape)
        # print(pred.shape)
        pred = F.log_softmax(pred, dim=1)
        gt = F.softmax(gt, dim=1)
        kl_loss = F.kl_div(pred, gt,reduction='batchmean')

        loss = kl_loss
        # delta = prediction.feature - batch["target"]["feature"]
        return self.cfg.weight * kl_loss
