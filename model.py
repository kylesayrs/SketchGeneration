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
        # convert to scale lower triangle
        scale_tril = torch.zeros((*sigmas.shape[:-1], 2, 2))
        first_indices, second_indices = torch.tril_indices(2, 2)
        scale_tril[:, :, :, first_indices, second_indices] = sigmas

        print(mus[0, 0])

        mixture = Categorical(logits=logits)
        components = MultivariateNormal(mus, scale_tril=scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        position_prev = torch.roll(position_true, 1, dims=1)
        position_prev[:, 0] = torch.tensor([0, 0])
        relative_positions_true = position_true - position_prev

        return -1 * mixture_model.log_prob(relative_positions_true).sum()


    def _get_pen_loss(
        self,
        pen_true: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        #print(pen_true)
        #print(pen_pred)
        return self.cross_entropy(
            pen_true.reshape(-1, pen_true.shape[-1]),
            pen_pred.reshape(-1, pen_pred.shape[-1])
        ).sum()
        

    def forward(
        self,
        xs: torch.Tensor,
        logits_pred: torch.Tensor,
        mus_pred: torch.Tensor,
        sigmas_pred: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        # unpack
        position_true, pen_true = torch.split(xs, [2, 3], dim=2)

        # compute separate losses
        position_loss = self._get_position_loss(position_true, logits_pred, mus_pred, sigmas_pred)
        pen_loss = self._get_pen_loss(pen_true, pen_pred)

        print(f"position_loss: {position_loss.item()}")
        print(f"pen_loss: {pen_loss.item()}")
        
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
        self.softmax = torch.nn.Softmax(dim=2)

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

        # TODO: experiment with adding a linear layer here

        logits_pred, mus_pred, sigmas_pred, pen_pred, _ = torch.split(ys, self._split_args, dim=2)
        logits_pred = self.softmax(logits_pred)
        mus_pred = self.sigmoid(mus_pred.reshape(*mus_pred.shape[:-1], self.model_config.num_components, -1))
        sigmas_pred = torch.exp(sigmas_pred.reshape(*sigmas_pred.shape[:-1], self.model_config.num_components, -1))
        pen_pred = self.softmax(pen_pred)

        return logits_pred, mus_pred, sigmas_pred, pen_pred
    
