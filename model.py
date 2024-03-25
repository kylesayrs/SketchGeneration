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
        self.pen_critic = torch.nn.NLLLoss(
            weight=torch.tensor([1.0, 2.0, 1.0]),
            reduction="mean"
        ) #torch.nn.CrossEntropyLoss()


    def _get_positions_loss(
        self,
        positions_true: torch.Tensor,
        is_end: torch.Tensor,
        logits: torch.Tensor,
        mus: torch.Tensor,
        sigmas_x: torch.Tensor,
        sigmas_y: torch.Tensor,
        sigmas_xy: torch.Tensor
    ) -> torch.Tensor:
        # create mixture model
        mixture_model = self.make_mixture_model(logits, mus, sigmas_x, sigmas_y, sigmas_xy)

        # compute true delta x and delta y
        positions_next = torch.roll(positions_true, -1, dims=1)
        deltas_true = positions_next - positions_true

        deltas_true[is_end] = 0.0  # by forcing all the positions at the
                                   # end to be zero, they will always be
                                   # correct and :. not contribute loss

        # mean negative log likelihood
        # original paper uses sum
        # then divided by max sequence length
        return -1 * mixture_model.log_prob(deltas_true).mean()


    def _get_pen_loss(
        self,
        pen_true: torch.Tensor,
        pen_pred: torch.Tensor
    ) -> torch.Tensor:
        # original paper uses sum of negative log loss here
        # then divided by max sequence length
        return self.pen_critic(
            pen_pred.flatten(0, 1),
            torch.argmax(pen_true.flatten(0, 1), dim=1)
        )
    

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
        scale_tril[:, :, :, 1, 0] = sigmas_xy

        # GMM
        mixture = Categorical(logits=logits)
        components = MultivariateNormal(mus, scale_tril=scale_tril)
        mixture_model = MixtureSameFamily(mixture, components)

        #print(logits[0])
        #print(mus[0, 0])
        #print(scale_tril[0, 0])

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

        # pen roll back to get what should be predicted
        # position rolling happens in _get_positions_loss
        # TODO: come back and simplify this
        pen_true = torch.roll(pen_true, -1, dims=1)
        pen_true[:, -1] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        is_end = pen_true[:, :, 2] == 1

        # compute separate losses
        position_loss = self._get_positions_loss(positions_true, is_end, logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy)
        pen_loss = self._get_pen_loss(pen_true, pen_pred)

        #print(positions_true[0, 10])
        #print(mus_pred[0, 10])
        #print(pen_true[0])
        #print(pen_pred[0])
        #print(torch.argmax(pen_true.reshape((-1, 3)), dim=1))
        #print(pen_loss)
        
        # sum losses
        return position_loss, pen_loss


class SketchDecoder(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config
    
        input_size = 5
        self.rnn = torch.nn.GRU(
            input_size,
            model_config.hidden_size,
            model_config.num_layers,
            dropout=model_config.dropout,
            bidirectional=False,
            batch_first=True
        )

        self.rnn2 = torch.nn.GRU(
            model_config.hidden_size,
            model_config.hidden_size,
            model_config.num_layers,
            dropout=model_config.dropout,
            bidirectional=False,
            batch_first=True
        )
        self.rnn3 = torch.nn.GRU(
            model_config.hidden_size,
            model_config.hidden_size,
            model_config.num_layers,
            dropout=model_config.dropout,
            bidirectional=False,
            batch_first=True
        )
        self.rnn4 = torch.nn.GRU(
            model_config.hidden_size,
            model_config.hidden_size,
            model_config.num_layers,
            dropout=model_config.dropout,
            bidirectional=False,
            batch_first=True
        )

        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p=model_config.dropout)
        self.layer_norm = torch.nn.LayerNorm(model_config.hidden_size)

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
        sigmas_x = self.elu(sigmas_x) + 1
        sigmas_y = self.elu(sigmas_y) + 1
        sigmas_xy = torch.tanh(sigmas_xy)

        # pen in 3 simplex
        pen_pred = self.softmax(pen_pred)

        return logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy, pen_pred


    def forward(
        self,
        xs: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor
        ]:
        # RNN layer
        ys, h_n = self.rnn(xs, h_0)
        ys = self.layer_norm(self.relu(self.dropout(ys)))
        h_n = self.layer_norm(self.relu(self.dropout(h_n)))

        ys, h_n = self.rnn2(ys, h_n)
        h_n = self.layer_norm(self.relu(self.dropout(ys)))
        ys = self.layer_norm(self.relu(self.dropout(h_n)))
        
        #ys, h_n = self.rnn3(ys, h_n)
        #ys = self.relu(ys)
        #h_n = self.relu(h_n)
        
        #ys, h_n = self.rnn4(ys, h_n)

        # linear layer
        ys = self.linear_0(ys)
        ys = self.relu(ys)
        # TODO: experiment with layernorm here
        ys = self.linear_1(ys)

        return self._unpack_outputs(ys), h_n
    
