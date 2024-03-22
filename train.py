import torch
import wandb
import numpy

from torch.utils.data import DataLoader

from data import load_drawings, pad_drawings
from config import TrainingConfig


def train():
    # set up configuration
    config = TrainingConfig()

    wandb.init(
        project="SketchGeneration",
        entity="kylesayrs",
        name=None,
        reinit=True,
        mode=config.wandb_mode,
        config=config.model_dump()
    )
    print(f"Run id: {wandb.run.id}")
    print(config)

    # load data
    drawings = load_drawings("data/flip flops.ndjson")
    drawings = pad_drawings(drawings, config.max_sequence_length)
    drawings = torch.tensor(drawings)
    print(f"Loaded {drawings.shape[0]} with sequence length {drawings.shape[1]}")


    """
    # split data
    train_drawings, test_drawings = split_data(drawings, test_ratio=0.2)

    # create datasets
    train_dataloader = DataLoader(train_drawings, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_drawings, batch_size=64, shuffle=True)

    # model and optimizer
    decoder = SketchDecoder(config.model_config)
    optimizer = 
    criterion = SketchCriterion()


    for epoch_index in range(config.num_epochs):
        for batch_index, xs in enumerate(train_dataset):
            hidden_states = torch.zeros(config.hidden_size)
            prev_xs = torch.zeros(xs.shape[0], xs.shape[2])

            for sequence_index in range(max_sequence_length):
                next_xs = xs[:, sequence_index]
            
                # forward
                optimizer.zero_grad()
                logits_pred, mus_pred, sigmas_pred, pen_pred = decoder(prev_xs, h_initials, next_xs)
                
                # compute loss
                loss = criterion(next_xs, logits_pred, mus_pred, sigmas_pred, pen_pred)
                
                # backwards
                loss.backwards()
                optimizer.step()

                # move to next sample
                prev_xs = next_xs
                
            
    """


if __name__ == "__main__":
    train()
