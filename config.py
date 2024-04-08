from typing import Union

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    embed_dims: int = Field(default=128)
    hidden_dims: int = Field(default=256)
    num_heads: int = Field(default=1)

    max_sequence_length: int = Field(default=85)

    num_layers: int = Field(default=1)
    dropout: float = Field(default=0.1)
    num_components: int = Field(default=10)
    elu_alpha: float = Field(default=1.0)


class TrainingConfig(BaseModel):
    # data
    num_epochs: int = Field(default=150)
    batch_size: int = Field(default=128)
    max_sequence_length: int = Field(default=85)
    data_sparsity: int = Field(default=1)
    aug_scale_factor: float = Field(default=0.1)

    # optimizer
    learning_rate: float = Field(default=5e-06)
    gradient_clip: Union[float, None] = Field(default=None)

    # logging
    wandb_mode: str = Field(default="online")
    log_frequency: int = Field(default=100)

    save_parent_dir: Union[str, None] = Field(default="checkpoints")
