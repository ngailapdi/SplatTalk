from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_mse_feat import LossMseFeat, LossMseFeatCfgWrapper
from .loss_cos import LossCos, LossCosCfgWrapper
from .loss_kl import LossKL, LossKLCfgWrapper
from .loss_gaussian import LossGaussian, LossGaussianCfgWrapper


LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossMseFeatCfgWrapper: LossMseFeat,
    LossCosCfgWrapper: LossCos,
    LossKLCfgWrapper: LossKL,
    LossGaussianCfgWrapper: LossGaussian

}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper | LossMseFeatCfgWrapper | LossCosCfgWrapper | LossKLCfgWrapper | LossGaussianCfgWrapper
# LossCfgWrapper =  LossMseFeatCfgWrapper | LossCosCfgWrapper



def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
