from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import numpy as np
import os

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
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


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        phase: int = 1,
        path: str = None,
    ):
        orig_num_views, _, _ = extrinsics.shape
        # print('Orig Views: ', orig_num_views)

        invalid_id = torch.unique(torch.where(~torch.isfinite(extrinsics))[0])
        # print('INVALID: ', invalid_id)
        if len(invalid_id) > 0:
            num_views = orig_num_views - len(invalid_id)
        else:
            num_views = orig_num_views
        # print('Views: ', num_views)

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
           # When testing, always use the full gap.
           max_gap = self.cfg.max_distance_between_context_views
           min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        if self.cfg.random:
            num_context_views = np.random.randint(2, self.num_context_views+1)
        else:
            num_context_views = self.num_context_views
        # print('NUM: ', num_context_views)
        if (num_context_views > (num_views-1) // context_gap + 1) and not self.cfg.random:
            num_context_views = (num_views-1)//context_gap+1
            # raise ValueError("Not enough views for the context views!")
        num_context_views = min(num_context_views, (num_views-1) // context_gap + 1)
        index_context_left = torch.randint(
                num_views if self.cameras_are_circular else num_views - context_gap*(num_context_views+phase-2),
                size=tuple(),
                device=device,
            ).item()

        index_context_right = index_context_left + num_views
        context_views = torch.linspace(index_context_left, \
            index_context_right, self.cfg.num_context_views+1, dtype=torch.int64) % num_views
        context_views = context_views[:-1]
        if num_views != orig_num_views:
            valid_views = torch.arange(orig_num_views)
            valid_views = torch.tensor([i for i in valid_views if i not in invalid_id], dtype=torch.int64)
            context_views = valid_views[context_views]
        
        index_target = []
        if num_context_views == 2:
            per_size = 4
        elif num_context_views == 3:
            per_size = 2
        else:
            per_size = 1

        if self.cfg.num_target_views == -1:
            index_target = torch.arange(num_views)
        else:

            index_target = torch.randint(
                0,
                num_views,
                size=(self.cfg.num_target_views,),
                device=device,
            )
            
        return (
            torch.tensor(context_views, dtype=torch.int64, device=device),
            index_target,
            0
        )
    
    
    def sample1(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        phase: int = 1,
        path: str = None,
    ):
        orig_num_views, _, _ = extrinsics.shape
        # print('Orig Views: ', orig_num_views)

        invalid_id = torch.unique(torch.where(~torch.isfinite(extrinsics))[0])
        # print('INVALID: ', invalid_id)
        if len(invalid_id) > 0:
            num_views = orig_num_views - len(invalid_id)
            # print('Orig: ', orig_num_views)
            # print('Views: ', num_views)

        else:
            num_views = orig_num_views

        # print('Views: ', num_views)

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
           # When testing, always use the full gap.
           max_gap = self.cfg.max_distance_between_context_views
           min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        if self.cfg.random:
            num_context_views = np.random.randint(2, self.num_context_views+1)
        else:
            num_context_views = self.num_context_views
        # print('NUM: ', num_context_views)
        if (num_context_views > (num_views-1) // context_gap + 1) and not self.cfg.random:
            num_context_views = (num_views-1)//context_gap+1
            # raise ValueError("Not enough views for the context views!")
        num_context_views = min(num_context_views, (num_views-1) // context_gap + 1)
        index_context_left = torch.randint(
                num_views if self.cameras_are_circular else num_views - context_gap*(num_context_views+phase-2),
                size=tuple(),
                device=device,
            ).item()
        
        index_target = []
        if num_context_views == 2:
            per_size = 4
        elif num_context_views == 3:
            per_size = 2
        else:
            per_size = 1
            
        context_views = [index_context_left % num_views]
        for i in range(num_context_views-1):
            index_context_right = context_views[i] + context_gap

            if self.is_overfitting:
                index_context_left *= 0
                index_context_right *= 0
                index_context_right += max_gap

            # Pick the target view indices.
            index_target.append(torch.randint(
                    context_views[i] + self.cfg.min_distance_to_context_views,
                    index_context_right - self.cfg.min_distance_to_context_views,
                    size=(per_size,),
                    device=device,
                )% num_views)
            context_views.append(index_context_right % num_views)

        index_context_left += context_gap
        

        index_target = torch.cat(index_target)
        if num_views != orig_num_views:
            valid_views = torch.arange(orig_num_views)
            valid_views = torch.tensor([i for i in valid_views if i not in invalid_id], dtype=torch.int64)
            # context_views = torch.tensor([i for i in context_views if i not in invalid_id], dtype=torch.int64)
            # index_target = torch.tensor([i for i in index_target if i not in invalid_id], dtype=torch.int64)
            # temp = torch.randint(0, len(valid_views), size=(20,), device=device)
            # print(temp)
            context_views = valid_views[context_views]
            # print(context_views)
            # print('VALID: ',extrinsics[context_views])

            index_target = valid_views[index_target]
            # print('SCENE: ',scene)
        # index_target = torch.randint(
        #     0,
        #     num_views,
        #     size=(self.cfg.num_target_views,),
        #     device=device,
        # )
        # index_target = context_views[:self.cfg.num_target_views]
        # print('context: ', context_views)
        # print('target: ',index_target)



        
        
        
        context_views = np.array(context_views)
        # if self.cameras_are_circular:
        #     context_views %= num_views

        context_views = context_views.tolist()
        return (
            torch.tensor(context_views, dtype=torch.int64, device=device),
            # torch.cat(index_target),
            torch.tensor(index_target, dtype=torch.int64, device=device),
            0
        )
       
        
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
