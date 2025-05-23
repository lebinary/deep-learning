import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from grader.metrics import DetectionMetric
from homework.utils import create_subset, dataloader_stats

from .models import ClassificationLoss, RegressionLoss, load_model, save_model
from .datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    sample_percent: float = 1,
    seg_weight: float = 1.0,  # Added weight for segmentation loss
    depth_weight: float = 1.0,  # Added weight for depth loss
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
    full_train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    full_val_data = load_data("road_data/val", shuffle=False)
    
    # Sample subsets based on percentages
    train_dataset = create_subset(full_train_data.dataset, sample_percent)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training on {len(train_dataset)} samples ({sample_percent * 100}% of full dataset)")
    
    val_dataset = create_subset(full_val_data.dataset, sample_percent)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validating on {len(val_dataset)} samples ({sample_percent * 100}% of full dataset)")

    # create loss function and optimizer
    seg_loss_fn = ClassificationLoss()
    depth_loss_fn = RegressionLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # Setup metrics for both training and validation
    train_metrics = DetectionMetric(num_classes=3)
    val_metrics = DetectionMetric(num_classes=3)

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_metrics.reset()
        val_metrics.reset()

        # TRAINING
        model.train()
        for batch in train_data:
            img, track, depth = batch['image'].to(device), batch['track'].to(device), batch['depth'].to(device)

            # Forward pass
            logits, raw_depth = model(img)

            # Compute losses
            seg_loss = seg_loss_fn(logits, track)
            depth_loss = depth_loss_fn(raw_depth, depth)
            total_loss = (seg_weight * seg_loss) + (depth_weight * depth_loss)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update metrics (after converting logits to predictions)
            with torch.no_grad():
                pred = logits.argmax(1)
                train_metrics.add(pred, track, raw_depth, depth)

        # VALIDATION
        model.eval()
        with torch.inference_mode():

            for batch in val_data:
                img, track, depth = batch['image'].to(device), batch['track'].to(device), batch['depth'].to(device)
                
                pred, pred_depth = model.predict(img)

                # Update metrics with predictions
                val_metrics.add(pred, track, pred_depth, depth)

        # log average train and val accuracy to tensorboard
        train_results = train_metrics.compute()
        val_results = val_metrics.compute()

        # Log to tensorboard
        for key, value in train_results.items():
            logger.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_results.items():
            logger.add_scalar(f'Val/{key}', value, epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d}/{num_epoch:2d}: "
                f"val_iou={val_results['iou']:.4f} "
                f"val_acc={val_results['accuracy']:.4f} "
                f"val_depth={val_results['abs_depth_error']:.4f} "
                f"val_tp_depth={val_results['tp_depth_error']:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    timestamp = datetime.now().strftime('%Y%m%d_%H%M') 
    save_model(model, timestamp)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--sample_percent", type=float, default=1.0)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--depth_weight", type=float, default=1.0)


    # pass all arguments to train
    train(**vars(parser.parse_args()))
