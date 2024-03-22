from typing import Tuple

import torch
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily
)

"""position_loss = self.position_critic(position_true, logits_pred, mus_pred, sigmas_pred)
    pen_loss = self.pen_critic(pen_true, pen_pred)

:return: _description_
"""


class PositionCritic(torch.nn.Module):
    def forward(position_true: torch.Tensor, logits: torch.Tensor, mus: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        scale_tril = torch.tensor([[sigmas[0], sigmas[2]], [0.0, sigmas[1]]])

        mixture = Categorical(logits=logits)
        components = MultivariateNormal(mus, scale_tril=scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        return mixture_model.log_prob(position_true)


class SketchDecoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_components: int = 1,
    ):
        super().__init__()
    
        input_size = 5
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout
        )

        self.position_critic = PositionCritic()
        self.pen_critic = torch.nn.CrossEntropyLoss()


    def forward(
        self,
        prev_x: torch.Tensor,
        prev_h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gru(prev_x, prev_h)

        logits_pred = logits[:]
        mus_pred = torch.Sigmoid(logits[:])
        sigmas_pred = torch.exp(logits[:])
        pen_pred = logits[:]

        return logits_pred, mus_pred, sigmas_pred, pen_pred
    
