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
class LossGaussianCfg:
    weight: float


@dataclass
class LossGaussianCfgWrapper:
    gaussian: LossGaussianCfg


class LossGaussian(Loss[LossGaussianCfg, LossGaussianCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        encoder_results: dict,
        global_step: int,
        dr_prediction: DecoderOutput | None = None,
    ) -> Float[Tensor, ""]:
        feat = gaussians[0].features
        updated_feat = prediction.updated_gaussian_feature
        # print('feat: ',feat.shape)
        # print(updated_feat.shape)
        loss = torch.nn.MSELoss()(updated_feat, feat)

        return self.cfg.weight * loss
