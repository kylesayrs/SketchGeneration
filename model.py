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
        sigmas_x: torch.Tensor,
        sigmas_y: torch.Tensor,
        sigmas_xy: torch.Tensor
    ) -> torch.Tensor:
        # convert to scale lower triangle
        scale_tril = torch.zeros((*sigmas_x.shape, 2, 2))
        scale_tril[:, :, :, 0, 0] = torch.clamp(sigmas_x, min=1e-6)
        scale_tril[:, :, :, 1, 1] = torch.clamp(sigmas_y, min=1e-6)
        scale_tril[:, :, :, 1, 0] = sigmas_xy

        mixture = Categorical(logits=logits)
        components = MultivariateNormal(mus, scale_tril=scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        position_prev = torch.roll(position_true, 1, dims=1)
        position_prev[:, 0] = torch.tensor([0, 0])
        relative_positions_true = position_true - position_prev

        return -1 * mixture_model.log_prob(relative_positions_true).mean()


    def _get_pen_loss(
        self,
        pen_true: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        return self.cross_entropy(
            pen_true.reshape(-1, pen_true.shape[-1]),
            pen_pred.reshape(-1, pen_pred.shape[-1])
        ).mean()
        

    def forward(
        self,
        xs: torch.Tensor,
        logits_pred: torch.Tensor,
        mus_pred: torch.Tensor,
        sigmas_x: torch.Tensor,
        sigmas_y: torch.Tensor,
        sigmas_xy: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        # unpack
        position_true, pen_true = torch.split(xs, [2, 3], dim=2)

        # compute separate losses
        position_loss = self._get_position_loss(position_true, logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy)
        pen_loss = self._get_pen_loss(pen_true, pen_pred)

        print("-----")
        print(pen_true[0, 3])
        print(position_true[0, 3])
        print("*****")
        print(pen_pred[0, 3])
        print(mus_pred[0, 0, 0])
        print(sigmas_x[0, 3])
        print(sigmas_y[0, 3])
        print(sigmas_xy[0, 3])
        print("-----")

        print(f"position_loss: {position_loss.item()}")
        print(f"pen_loss: {pen_loss.item()}")
        
        # sum losses
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

        self.output_size = 6 * model_config.num_components + 3

        self.linear_0 = torch.nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear_1 = torch.nn.Linear(model_config.hidden_size, self.output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=2)

        self.to(torch.float32)


    @cached_property
    def _split_args(self) -> List[Tuple[int, int]]:
        return [
            self.model_config.num_components,
            2 * self.model_config.num_components,
            self.model_config.num_components,
            self.model_config.num_components,
            self.model_config.num_components,
            3
        ]
    

    def _unpack_outputs(self, ys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy, pen_pred = torch.split(ys, self._split_args, dim=2)

        # logits in K simplex
        logits_pred = self.softmax(logits_pred)
        
        # means in [-1, 1]
        mus_pred = torch.tanh(mus_pred.reshape(*mus_pred.shape[:-1], self.model_config.num_components, -1))

        # diagonal sigmas in [0, 1]
        sigmas_x = torch.sigmoid(sigmas_x)
        sigmas_y = torch.sigmoid(sigmas_y)

        # covariance sigmas in [-1, 1]
        sigmas_xy = torch.tanh(sigmas_xy)

        # pen in 3 simplex
        pen_pred = self.softmax(pen_pred)

        return logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy, pen_pred


    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # RNN layer
        ys, _hidden_state = self.gru(xs)

        # linear layer
        #ys = self.linear_0(ys)
        #ys = self.relu(ys)
        ys = self.linear_1(ys)

        return self._unpack_outputs(ys)
    
