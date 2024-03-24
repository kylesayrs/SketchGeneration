from typing import Union

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    hidden_size: int = Field(default=128)
    num_layers: int = Field(default=1)
    dropout: float = Field(default=0.1)
    num_components: int = Field(default=15)
    sigma_min: float = Field(default=1e-6)


class TrainingConfig(BaseModel):
    # data
    num_epochs: int = Field(default=10)
    batch_size: int = Field(default=64)
    max_sequence_length: int = Field(default=100)
    data_sparsity: int = Field(default=1)

    # optimizer
    learning_rate: float = Field(default=1e-5)
    gradient_clip: float = Field(default=0.12)

    # logging
    wandb_mode: str = Field(default="online")
    log_frequency: int = Field(default=100)

    save_parent_dir: Union[str, None] = Field(default="checkpoints")
