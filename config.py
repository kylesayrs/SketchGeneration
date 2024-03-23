from typing import Union

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    hidden_size: int = Field(default=128)
    num_layers: int = Field(default=3)
    dropout: float = Field(default=0.05)
    num_components: int = Field(default=1)
    sigma_min: float = Field(default=1e-6)


class TrainingConfig(BaseModel):
    # data
    num_epochs: int = Field(default=10)
    batch_size: int = Field(default=64)
    max_sequence_length: int = Field(default=100)
    data_sparsity: int = Field(default=1)

    # optimizer
    learning_rate: float = Field(default=5e-4)

    # logging
    wandb_mode: str = Field(default="disabled")
    log_frequency: int = Field(default=100)

    save_parent_dir: Union[str, None] = Field(default="checkpoints")
