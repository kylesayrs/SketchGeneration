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
        reinit=True,
        mode=config.wandb_mode,
        config=config.model_dump()
    )
    print(f"Run id: {wandb.run.id}")
    print(config)

    # load data
    drawings = load_drawings("data/flip flops.ndjson", config.data_sparsity)
    drawings = pad_drawings(drawings, config.max_sequence_length)
    drawings = torch.tensor(drawings, dtype=torch.float32)

    # TESTING
    drawings *= 0
    drawings[:, :, 4] = 1
    
    print(f"Loaded {drawings.shape[0]} with sequence length {drawings.shape[1]}")

    # split data
    train_drawings, test_drawings = train_test_split(drawings, train_size=0.8)

    # create datasets
    train_dataloader = DataLoader(train_drawings, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_drawings, batch_size=config.batch_size, shuffle=True)

    # model and optimizer
    decoder = SketchDecoder(model_config)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate)
    criterion = SketchCritic()

    for epoch_index in range(config.num_epochs):
        for batch_index, samples in enumerate(train_dataloader):            
            # forward
            optimizer.zero_grad()
            logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy, pen_pred = decoder(samples)

            # compute loss
            loss = criterion(samples, logits_pred, mus_pred, sigmas_x, sigmas_y, sigmas_xy, pen_pred)
            print(f"loss: {loss.item()}")

            # backwards
            loss.backward()
            optimizer.step()
    

if __name__ == "__main__":
    train()
