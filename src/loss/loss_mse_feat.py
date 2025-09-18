from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch.nn.functional as F 


@dataclass
class LossMseFeatCfg:
    weight: float


@dataclass
class LossMseFeatCfgWrapper:
    mse_feat: LossMseFeatCfg


class LossMseFeat(Loss[LossMseFeatCfg, LossMseFeatCfgWrapper]):
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
        l1_loss = torch.nn.MSELoss()(pred, gt)
        # cos_loss = 1 - torch.nn.functional.cosine_similarity(pred, gt).mean()
        loss = l1_loss
        # delta = prediction.feature - batch["target"]["feature"]
        return self.cfg.weight * loss
