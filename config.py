from typing import Union

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    embed_dims: int = Field(default=32)
    hidden_dims: int = Field(default=64)
    num_heads: int = Field(default=1)

    max_sequence_length: int = Field(default=100)

    num_layers: int = Field(default=4)
    dropout: float = Field(default=0.0)
    num_components: int = Field(default=1)


class TrainingConfig(BaseModel):
    # data
    num_epochs: int = Field(default=10_000)
    batch_size: int = Field(default=128)
    max_sequence_length: int = Field(default=85)
    data_sparsity: int = Field(default=1)
    aug_scale_factor: float = Field(default=0.1)

    # optimizer
    learning_rate: float = Field(default=2e-4)
    gradient_clip: Union[float, None] = Field(default=10_000)

    # logging
    wandb_mode: str = Field(default="disabled")
    log_frequency: int = Field(default=100)

    save_parent_dir: Union[str, None] = Field(default="checkpoints")
