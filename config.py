from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    hidden_size: int = Field(default=128)


class TrainingConfig(BaseModel):
    num_epochs: int = Field(default=10)
    max_sequence_length: int = Field(default=75)

    wandb_mode: str = Field(default="disabled")
