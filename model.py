from typing import List, Tuple

import torch
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily
)
from functools import cache

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

        return mixture_model.log_prob(position_true)


    def _get_pen_loss(
        self,
        pen_true: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        return self.cross_entropy(pen_true, pen_pred)
        

    def forward(
        self,
        x_true: torch.Tensor,
        logits_pred: torch.Tensor,
        mus_pred: torch.Tensor,
        sigmas_pred: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        # unpack
        position_true, pen_true = torch.tensor_split(x_true, 3)

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


    @cache
    def _get_output_splits(self) -> List[Tuple[int, int]]:
        logits_size = self.model_config.num_components
        logits_end = 0 + logits_size

        mus_size = 2 * self.model_config.num_components
        mus_end = logits_end + mus_size

        sigmas_size = 3 * self.model_config.num_components
        sigmas_end = mus_end + sigmas_size
        
        pen_size = 3
        pen_end = sigmas_end + pen_size


        return [
            (0, logits_end),
            (logits_end, mus_end),
            (mus_end, sigmas_end),
            (sigmas_end, pen_end)
        ]


    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ys, _hidden_state = self.gru(xs)
        print(f"ys: {ys.shape}")

        # TODO: experiment with adding a linear layer here

        splits = self._get_output_splits()
        
        logits_pred = self.sigmoid(ys[:, :, splits[0][0]: splits[0][1]])
        mus_pred    = self.sigmoid(ys[:, :, splits[1][0]: splits[1][1]])
        sigmas_pred = torch.exp(   ys[:, :, splits[2][0]: splits[2][1]])
        pen_pred    = self.sigmoid(ys[:, :, splits[3][0]: splits[3][1]])
        print(f"logits_pred: {logits_pred.shape}")
        print(f"mus_pred: {mus_pred.shape}")
        print(f"sigmas_pred: {sigmas_pred.shape}")
        print(f"pen_pred: {pen_pred.shape}")

        return logits_pred, mus_pred, sigmas_pred, pen_pred
    
