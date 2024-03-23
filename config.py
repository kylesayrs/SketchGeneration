from typing import Union

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    hidden_size: int = Field(default=128)
    num_layers: int = Field(default=3)
    dropout: float = Field(default=0.3)
    num_components: int = Field(default=1)
    bidirectional: bool = Field(default=False)

    @model_validator(mode="after")
    def check_passwords_match(self) -> "ModelConfig":
        self.hidden_size >= 5 * self.num_components + 3
        return self


class TrainingConfig(BaseModel):
    # data
    num_epochs: int = Field(default=10)
    batch_size: int = Field(default=64)
    max_sequence_length: int = Field(default=75)
    data_sparsity: int = Field(default=10)

    # optimizer
    learning_rate: float = Field(default=1e-4)

    # logging
    wandb_mode: str = Field(default="online")
    log_frequency: int = Field(default=100)

    save_parent_dir: Union[str, None] = Field(default="checkpoints")
