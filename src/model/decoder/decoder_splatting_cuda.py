from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput
import torch.nn.functional as F
import copy


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]



class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        no_color: bool = False,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        if not no_color:
            # updated_gaussians = self.viewdependent(gaussians, extrinsics)
            updated_gaussians = gaussians.features

            color, depth, language = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                repeat(updated_gaussians, "b g c -> (b v) g c", v=v)
            )
            color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
            depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)
            depth = depth / 2

            language = rearrange(language, "(b v) c h w -> b v c h w", b=b, v=v)
        
        else:
            color = None

        return DecoderOutput(
            color,
            None
            if depth_mode is None
            else depth,
            language,
            updated_gaussians
        )

    def decode_feat(self, x):
        for m in self.feat_decoder:
            x = m(x)    
        # x = x / x.norm(dim=-1, keepdim=True)
        return x

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)