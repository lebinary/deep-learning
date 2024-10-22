import torch
from torch.utils.data import DataLoader, Dataset

def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Arguments:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels

    Returns:
        a single torch.Tensor scalar
    """
    outputs_idx = outputs.max(1)[1].type_as(labels)

    return (outputs_idx == labels).float().mean()


def dataloader_stats(data_loader: DataLoader | Dataset):
    # Basic stats
    print(f"Number of batches: {len(data_loader)}")
    print(f"Batch size: {data_loader.batch_size}")
    print(f"Total samples: {len(data_loader.dataset)}")

    # Get one batch to see shapes and data types
    batch = next(iter(data_loader))
    print("\nBatch content:")
    if isinstance(batch, list):
        print(f"  Shape: {batch[0].shape}")
        print(f"  Type: {batch[0].dtype}")    
    else:
        for key in batch:
            print(f"{key}:")
            print(f"  Shape: {batch[key].shape}")
            print(f"  Type: {batch[key].dtype}")


# Function to create subset based on percentage
def create_subset(dataset, percent):
    if percent <= 0 or percent > 1:
        raise ValueError("Percentage must be between 0 and 1")
    num_samples = int(len(dataset) * percent)
    indices = torch.randperm(len(dataset))[:num_samples]
    return torch.utils.data.Subset(dataset, indices)    