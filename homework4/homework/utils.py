import torch

def create_subset(dataset, percent):
    if percent <= 0 or percent > 1:
        raise ValueError("Percentage must be between 0 and 1")
    num_samples = int(len(dataset) * percent)
    indices = torch.randperm(len(dataset))[:num_samples]
    return torch.utils.data.Subset(dataset, indices)    