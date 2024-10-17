import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .datasets.classification_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    sample_percent: float = 1,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps") # for Arm Macs
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()
    
    # Load full datasets
    full_train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    full_val_data = load_data("classification_data/val", shuffle=False)
    
    # Sample subsets based on percentages
    train_dataset = create_subset(full_train_data.dataset, sample_percent)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training on {len(train_dataset)} samples ({sample_percent}% of full dataset)")
    
    val_dataset = create_subset(full_val_data.dataset, sample_percent)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validating on {len(val_dataset)} samples ({sample_percent}% of full dataset)")

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = ...
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # Forward pass
            logits = model(img)
            loss = loss_func(logits, label)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute training accuracy
            pred = logits.argmax(dim=1)
            acc = (pred == label).float().mean().item()
            metrics["train_acc"].append(acc)
            
            # Log training loss
            logger.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # TODO: compute validation accuracy
                logits = model(img)
                pred = logits.argmax(dim=1)
                acc = (pred == label).float().mean().item()
                metrics["val_acc"].append(acc)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        logger.add_scalar('Accuracy/val', epoch_val_acc, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

# Function to create subset based on percentage
def create_subset(dataset, percent):
    if percent <= 0 or percent > 1:
        raise ValueError("Percentage must be between 0 and 1")
    num_samples = int(len(dataset) * percent)
    indices = torch.randperm(len(dataset))[:num_samples]
    return torch.utils.data.Subset(dataset, indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--sample_percent", type=float, default=1.0)


    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=4)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
