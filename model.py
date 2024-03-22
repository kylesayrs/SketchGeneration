from typing import List, Tuple

import torch
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily
)
from functools import cache, cached_property

from config import ModelConfig

"""position_loss = self.position_critic(position_true, logits_pred, mus_pred, sigmas_pred)
    pen_loss = self.pen_critic(pen_true, pen_pred)

:return: _description_
"""


class SketchCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cross_entropy = torch.nn.CrossEntropyLoss()


    def _get_position_loss(
        self,
        position_true: torch.Tensor,
        logits: torch.Tensor,
        mus: torch.Tensor,
        sigmas: torch.Tensor
    ) -> torch.Tensor:
        scale_tril = torch.tensor([[sigmas[:, 0], sigmas[:, 2]], [0.0, sigmas[:, 1]]])

        mixture = Categorical(logits=logits)
        components = MultivariateNormal(mus, scale_tril=scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        # TODO: need to convert positions to relative positions

        return mixture_model.log_prob(position_true)


    def _get_pen_loss(
        self,
        pen_true: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        return self.cross_entropy(pen_true, pen_pred)
        

    def forward(
        self,
        xs: torch.Tensor,
        logits_pred: torch.Tensor,
        mus_pred: torch.Tensor,
        sigmas_pred: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        print(f"xs: {xs.shape}")
        # unpack
        asdf = torch.tensor_split(xs, [2], dim=2)
        print(len(asdf))
        print(asdf[0].shape)
        print(asdf[1].shape)
        print(asdf[2].shape)
        exit(0)
        print(f"position_true: {position_true.shape}")
        print(f"pen_true: {pen_true.shape}")
        exit(0)

        # compute separate losses
        position_loss = self._get_position_loss(position_true, logits_pred, mus_pred, sigmas_pred)
        pen_loss = self._get_pen_loss(pen_true, pen_pred)
        
        # mix losses
        return position_loss + pen_loss


class SketchDecoder(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        print(model_config)
        super().__init__()

        self.model_config = model_config
    
        input_size = 5
        self.gru = torch.nn.GRU(
            input_size,
            model_config.hidden_size,
            model_config.num_layers,
            bidirectional=model_config.bidirectional,
            dropout=model_config.dropout,
            batch_first=True
        )

        self.sigmoid = torch.nn.Sigmoid()

        self.to(torch.float32)


    @cached_property
    def _split_args(self, ) -> List[Tuple[int, int]]:
        return [
            self.model_config.num_components,
            2 * self.model_config.num_components,
            3 * self.model_config.num_components,
            3,
            self.model_config.hidden_size - 6 * self.model_config.num_components - 3,
        ]


    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ys, _hidden_state = self.gru(xs)
        print(f"ys: {ys.shape}")

        # TODO: experiment with adding a linear layer here

        logits_pred, mus_pred, sigmas_pred, pen_pred, _ = torch.split(ys, self._split_args, dim=2)
        
        print(f"logits_pred: {logits_pred.shape}")
        print(f"mus_pred: {mus_pred.shape}")
        print(f"sigmas_pred: {sigmas_pred.shape}")
        print(f"pen_pred: {pen_pred.shape}")

        return logits_pred, mus_pred, sigmas_pred, pen_pred
    
