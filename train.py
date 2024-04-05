import os
import json
import torch
import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import SketchCritic, SketchDecoder
from config import TrainingConfig, ModelConfig
from data import load_drawings, pad_drawings, DrawingsDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    # set up configuration
    config = TrainingConfig()
    model_config = ModelConfig()

    wandb.init(
        project="SketchGeneration",
        entity="kylesayrs",
        name=None,
        reinit=False,
        mode=config.wandb_mode,
        config=config.model_dump().update(model_config.model_dump())
    )
    print(f"Run id: {wandb.run.id}")
    print(config)

    # load data
    drawings = load_drawings("data/moon.ndjson", config.data_sparsity)
    drawings = pad_drawings(drawings, config.max_sequence_length)
    drawings = torch.tensor(drawings, dtype=torch.float32, device=DEVICE)

    # Toy dataset
    """
    drawings = torch.tensor([
        [0.0, 0.0, 1, 0, 0],
        [0.1, 0.0, 1, 0, 0],
        [0.2, 0.0, 1, 0, 0],
        [0.3, 0.0, 1, 0, 0],
        [0.4, 0.0, 1, 0, 0],
        [0.5, 0.0, 1, 0, 0],
        [0.5, 0.0, 0, 1, 0],
        [0.4, 0.1, 1, 0, 0],
        [0.3, 0.2, 1, 0, 0],
        [0.2, 0.3, 1, 0, 0],
        [0.1, 0.4, 1, 0, 0],
        [0.0, 0.5, 0, 1, 0],
        [0.0, 0.0, 0, 0, 1],
        [0.0, 0.0, 0, 0, 1],
    ], dtype=torch.float32).repeat(200_000, 1, 1)
    """
    
    print(f"Loaded {drawings.shape[0]} with sequence length {drawings.shape[1]}")

    # split data
    train_drawings, test_drawings = train_test_split(drawings, train_size=0.8)

    # create datasets
    train_dataset = DrawingsDataset(train_drawings, scale_factor=config.aug_scale_factor)
    test_dataset = DrawingsDataset(test_drawings, scale_factor=None)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # model and optimizer
    model = SketchDecoder(model_config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = SketchCritic(sigma_min=model_config.sigma_min).to(DEVICE)

    # cache save dir
    save_dir = (
        os.path.join(config.save_parent_dir, wandb.run.id)
        if config.save_parent_dir is not None
        else None
    )

    # begin training
    position_losses = []
    pen_losses = []
    losses = []
    total_num_samples = 0
    max_gradient_norm = 0
    for epoch_index in range(config.num_epochs):
        for batch_index, (samples) in enumerate(train_dataloader):
            # forward
            model.train()
            optimizer.zero_grad()
            outputs = model(samples)
            total_num_samples += len(samples)

            # compute loss
            position_loss, pen_loss = criterion(samples, *outputs)
            loss = pen_loss + position_loss

            # upload log
            position_losses.append(position_loss.item())
            pen_losses.append(pen_loss.item())
            losses.append(loss.item())

            # backwards
            loss.backward()

            # log maximum gradient
            with torch.no_grad():
                max_gradient_norm = max(
                    max_gradient_norm,
                    max(
                        parameter.grad.norm().item()
                        for parameter in model.parameters()
                    )
                )

            # optimize with gradient clipping
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()

            # test and log
            if batch_index % config.log_frequency == 0:
                with torch.no_grad():
                    test_samples = next(iter(test_dataloader))

                    model.eval()
                    test_outputs = model(test_samples)
                    test_position_loss, test_pen_loss = criterion(test_samples, *test_outputs)
                    
                # compute metrics and reset
                metrics = {
                    "total_num_samples": total_num_samples,
                    "train_position_loss": sum(position_losses) / len(position_losses),
                    "train_pen_loss": sum(pen_losses) / len(pen_losses),
                    "train_loss": sum(losses) / len(losses),
                    "test_position_loss": test_position_loss.item(),
                    "test_pen_loss": test_pen_loss.item(),
                    "test_loss": test_position_loss.item() + test_pen_loss.item(),
                    "max_gradient_norm": max_gradient_norm,
                }
                position_losses = []
                pen_losses = []
                losses = []
                max_gradient_norm = 0

                # log metrics
                print(f"[{epoch_index:04d}, {batch_index:04d}]: {json.dumps(metrics, indent=4)}")
                wandb.log(metrics)

                if config.save_parent_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)

                    # save model
                    file_name = f"{epoch_index:04d}_{batch_index:04d}.pth"
                    torch.save(model.state_dict(), os.path.join(save_dir, file_name))


if __name__ == "__main__":
    train()
