from typing import List, Tuple, Optional

import torch
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily
)
from functools import cached_property

from config import ModelConfig


class SketchCritic(torch.nn.Module):
    def __init__(self, sigma_min: float = 1e-2) -> None:
        super().__init__()

        self.sigma_min = sigma_min
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.has_been_below_0 = False


    def _get_position_loss(
        self,
        positions_true: torch.Tensor,
        logits: torch.Tensor,
        mus: torch.Tensor,
        sigmas_x: torch.Tensor,
        sigmas_y: torch.Tensor,
        sigmas_xy: torch.Tensor
    ) -> torch.Tensor:
        # create mixture model
        mixture_model = self.make_mixture_model(logits, mus, sigmas_x, sigmas_y, sigmas_xy)

        # compute true delta x and delta y
        # note: since first is [0, 0], last is [0, 0] after roll (good)
        positions_next = torch.roll(positions_true, -1, dims=1)
        relative_positions_true = positions_next - positions_true

        # mean negative log likelihood
        loss = -1 * mixture_model.log_prob(relative_positions_true).mean()
        if loss > 0:
            if self.has_been_below_0:
                print("chonko")
                print(positions_true[0])
                print(relative_positions_true[0])
                print(mixture_model.log_prob(relative_positions_true)[0])
                exit(0)

        else:
            self.has_been_below_0 = True

        return loss


    def _get_pen_loss(
        self,
        pen_true: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        return self.cross_entropy(
            pen_pred.reshape(-1, pen_pred.shape[-1]),
            pen_true.reshape(-1, pen_true.shape[-1])
        ).mean()
    

    def make_mixture_model(
        self,
        logits: torch.Tensor,
        mus: torch.Tensor,
        sigmas_x: torch.Tensor,
        sigmas_y: torch.Tensor,
        sigmas_xy: torch.Tensor
    ) -> torch.nn.Module:
        # convert to scale lower triangle
        scale_tril = torch.zeros((*sigmas_x.shape, 2, 2))
        scale_tril[:, :, :, 0, 0] = torch.clamp(sigmas_x, min=self.sigma_min)
        scale_tril[:, :, :, 1, 1] = torch.clamp(sigmas_y, min=self.sigma_min)
        scale_tril[:, :, :, 1, 0] = torch.clamp(sigmas_xy, min=-abs(self.sigma_min), max=abs(self.sigma_min))

        # GMM
        mixture = Categorical(logits=logits)
        components = MultivariateNormal(mus, scale_tril=scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        return mixture_model
        

    def forward(
        self,
        xs: torch.Tensor,
        logits_pred: torch.Tensor,
        mus_pred: torch.Tensor,
        sigmas_x: torch.Tensor,
        sigmas_y: torch.Tensor,
        sigmas_xy: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # unpack
        positions_true, pen_true = torch.split(xs, [2, 3], dim=2)

        # compute separate losses
        position_loss = self._get_position_loss(positions_true, logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy)
        pen_loss = self._get_pen_loss(pen_true, pen_pred)
        
        # sum losses
        return position_loss, pen_loss


class SketchDecoder(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config
    
        input_size = 5
        self.gru = torch.nn.GRU(
            input_size,
            model_config.hidden_size,
            model_config.num_layers,
            dropout=model_config.dropout,
            bidirectional=False,
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
        mus_pred = mus_pred.reshape(*mus_pred.shape[:-1], self.model_config.num_components, -1)

        # diagonal sigmas in [0, inf]
        # covariance sigmas in [-1, 1]
        sigmas_x = torch.exp(sigmas_x)
        sigmas_y = torch.exp(sigmas_y)
        sigmas_xy = torch.tanh(sigmas_xy)

        # pen in 3 simplex
        pen_pred = self.softmax(pen_pred)

        return logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy, pen_pred


    def forward(
        self,
        xs: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # RNN layer
        ys, hidden_state = self.gru(xs, h_0)

        # linear layer
        ys = self.linear_0(ys)
        ys = self.relu(ys)
        ys = self.linear_1(ys)

        return self._unpack_outputs(ys), hidden_state
    
