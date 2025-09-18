import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ...evaluation.evaluation_index_generator import IndexEntry
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerEvaluationCfg:
    # name: Literal["evaluation"]
    # index_path: Path
    # num_context_views: int

    name: Literal["evaluation"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    random: bool = False
    extra: bool = False


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        path: str = None,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Arbitrarily sample context and target views."""
        num_views, _, _ = extrinsics.shape

        index_context = torch.randint(
            0,
            num_views,
            size=(self.cfg.num_context_views,),
            device=device,
        )

        index_target = torch.randint(
            0,
            num_views,
            size=(self.cfg.num_target_views,),
            device=device,
        )

        return index_context, index_target, 0

    def sample_orig(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        phase: int = 1,
        path: str = None,
    ): 
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)
        try:
            target_indices = torch.cat([target_indices, torch.tensor(entry.extrapolation, dtype=torch.int64, device=device)])
            fvs_length = len(entry.extrapolation)
        except:
            fvs_length = 0
        return context_indices, target_indices, fvs_length

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
