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
class LossCosCfg:
    weight: float


@dataclass
class LossCosCfgWrapper:
    cos: LossCosCfg


class LossCos(Loss[LossCosCfg, LossCosCfgWrapper]):
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

        cos_loss = 1 - torch.nn.functional.cosine_similarity(pred, gt).mean()
        # cos_loss = torch.acos(torch.clamp(torch.nn.functional.cosine_similarity(pred, gt), -1.0, 1.0)).mean()

        loss = cos_loss
        # delta = prediction.feature - batch["target"]["feature"]
        return self.cfg.weight * loss
