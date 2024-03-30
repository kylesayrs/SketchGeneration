import os
import json
import torch
import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import TrainingConfig, ModelConfig
from data import load_drawings, pad_drawings
from model import SketchCritic, SketchDecoder


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
    drawings = load_drawings("data/flip flops.ndjson", config.data_sparsity)
    drawings = pad_drawings(drawings, config.max_sequence_length)
    drawings = torch.tensor(drawings, dtype=torch.float32)

    # TODO: create dataset and add data augmentation

    # TESTING
    #drawings[:, :, 0] = 0
    #drawings[:, :, 1] = 0
    #drawings[:, :, 2] = 0
    #drawings[:, :, 3] = 0
    #drawings[:, :, 4] = 1

    # TESTING
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
    train_dataloader = DataLoader(train_drawings, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_drawings, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # model and optimizer
    decoder = SketchDecoder(model_config)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate)
    criterion = SketchCritic(sigma_min=model_config.sigma_min)

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
    max_magnitude = 0
    total_num_samples = 0
    for epoch_index in range(config.num_epochs):
        for batch_index, samples in enumerate(train_dataloader):            
            # forward
            decoder.train()
            optimizer.zero_grad()
            outputs = decoder(samples)
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

            # optimize with gradient clipping
            with torch.no_grad():
                max_magnitude = max(max_magnitude, max(p.abs().max() for p in decoder.parameters()))
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=config.gradient_clip)
            optimizer.step()

            # test and log
            if batch_index % config.log_frequency == 0:
                with torch.no_grad():
                    test_samples = next(iter(test_dataloader))

                    decoder.eval()
                    test_outputs = decoder(test_samples)
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
                    "max_magnitude": max_magnitude.item()
                }
                position_losses = []
                pen_losses = []
                losses = []
                max_magnitude = 0

                # log metrics
                print(f"[{epoch_index:04d}, {batch_index:04d}]: {json.dumps(metrics, indent=4)}")
                wandb.log(metrics)

                if config.save_parent_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)

                    # save model
                    file_name = f"{epoch_index:04d}_{batch_index:04d}.pth"
                    torch.save(decoder.state_dict(), os.path.join(save_dir, file_name))


if __name__ == "__main__":
    train()
