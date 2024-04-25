from typing import List, Tuple, Optional

import math
import torch
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily
)
from functools import cached_property

from config import ModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(torch.nn.Module):
    """
    Batch-first variant of torch's Positional Encoding
    """

    def __init__(self, embed_dims: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        positions = torch.arange(max_len).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, embed_dims, 2) * (-math.log(10000.0) / embed_dims))
        positions = positions * division_term

        positional_encoding = torch.zeros(1, max_len, embed_dims, device=DEVICE)
        positional_encoding[0, :, 0::2] = torch.sin(positions)
        positional_encoding[0, :, 1::2] = torch.cos(positions)
        self.register_buffer("positional_encoding", positional_encoding)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.positional_encoding[:, :x.size(1)]
        return self.dropout(x)


class SketchCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.pen_critic = torch.nn.NLLLoss(
            weight=torch.tensor([1.0, 1.0, 1.0]),
            reduction="mean"
        )


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
        scale_tril = torch.zeros((*sigmas_x.shape, 2, 2), device=DEVICE)
        scale_tril[:, :, :, 0, 0] = sigmas_x
        scale_tril[:, :, :, 1, 1] = sigmas_y
        scale_tril[:, :, :, 1, 0] = sigmas_xy

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

        # pen roll back to get what should be predicted
        # position rolling happens in _get_positions_loss
        # TODO: come back and simplify this
        pen_true = torch.roll(pen_true, -1, dims=1)
        pen_true[:, -1] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        is_end = pen_true[:, :, 2] == 1

        # compute separate losses
        position_loss = self._get_positions_loss(positions_true, is_end, logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy)
        pen_loss = self._get_pen_loss(pen_true, pen_pred)
        
        # sum losses
        return position_loss, pen_loss


class SketchDecoder(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config
    
        input_size = 5
        self.tokenizer = torch.nn.Linear(input_size, model_config.embed_dims, device=DEVICE)
        self.positional_encoder = PositionalEncoding(model_config.embed_dims, dropout=model_config.dropout, max_len=model_config.max_sequence_length)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_config.embed_dims,
            nhead=model_config.num_heads,
            dim_feedforward=model_config.hidden_dims,
            dropout=model_config.dropout,
            activation=torch.nn.functional.relu,
            dtype=torch.float32,
            batch_first=True,
            device=DEVICE
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            model_config.num_layers,
            enable_nested_tensor=False,
        )

        self.output_size = 6 * model_config.num_components + 3
        self.linear_0 = torch.nn.Linear(model_config.embed_dims, self.output_size, device=DEVICE)

        self.elu = torch.nn.ELU(alpha=model_config.elu_alpha)
        self.softmax = torch.nn.Softmax(dim=2)

        self.init_weights()
        self.to(dtype=torch.float32, device=DEVICE)


    def init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.data.dim() == 2:
                torch.nn.init.kaiming_uniform_(param)
        

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
        sigmas_x = self.elu(sigmas_x) + self.elu.alpha + 0.01
        sigmas_y = self.elu(sigmas_y) + self.elu.alpha + 0.01
        sigmas_xy = torch.tanh(sigmas_xy) + 0.01

        # pen in 3 simplex
        pen_pred = self.softmax(pen_pred)

        return logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy, pen_pred


    def forward(self, xs: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # tokenizer
        xs = self.tokenizer(xs)
        xs = self.positional_encoder(xs)

        # decoder
        mask = torch.nn.Transformer.generate_square_subsequent_mask(xs.shape[1]).to(DEVICE)
        xs = self.transformer(xs, mask=mask)

        # linear layer
        ys = self.linear_0(xs)

        return self._unpack_outputs(ys)
    
