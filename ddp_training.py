"""
Distributed Data Parallel (DDP) PyTorch Training Script

This script demonstrates distributed data-parallel training in PyTorch using
multiple GPUs. Based on the tutorial at:
https://sebastianraschka.com/teaching/pytorch-1h/

Key concepts covered:
- DistributedDataParallel (DDP) for multi-GPU training
- Process group initialization and cleanup
- Distributed sampling to ensure non-overlapping data across GPUs
- Gradient synchronization across devices

Usage:
    # For 2 GPUs:
    torchrun --nproc_per_node=2 ddp_training.py
    # For all available GPUs:
    torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) ddp_training.py

Note: This script requires multiple GPUs to run effectively. For single GPU
or CPU-only systems, use the regular training script instead.
"""

import os
import platform

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def ddp_setup(rank, world_size):
    """
    Initialize a distributed process group for DDP training.

    Args:
        rank: Unique process ID (GPU index)
        world_size: Total number of processes (GPUs) in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # Initialize process group with appropriate backend
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library (optimal for GPU-to-GPU)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set the device for the current process
    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    """
    Simple dataset implementation for demonstration purposes.

    This dataset can be easily replaced with real-world datasets
    like ImageNet, CIFAR, or custom datasets.
    """

    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    """
    Multi-layer perceptron with 2 hidden layers.

    This simple architecture demonstrates the DDP concepts.
    For real applications, this could be replaced with ResNet,
    Transformer, or other complex architectures.
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # Output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


def prepare_dataset():
    """
    Create training and test datasets with data loaders.

    For DDP, the key component is the DistributedSampler which ensures
    that each GPU processes different, non-overlapping data samples.
    """
    # Create toy dataset - small for demonstration
    X_train = torch.tensor(
        [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
    )
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase dataset size for more GPUs (up to 8)
    # This creates synthetic variations of the original data
    factor = 4
    X_train = torch.cat(
        [X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)]
    )
    y_train = y_train.repeat(factor)
    X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    y_test = y_test.repeat(factor)

    # Create dataset objects
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    # Create data loaders with DistributedSampler for training
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # Must be False when using DistributedSampler
        pin_memory=True,  # Speeds up CPU-to-GPU transfer
        drop_last=True,  # Ensures consistent batch sizes across GPUs
        sampler=DistributedSampler(train_ds),  # Key component for DDP
    )

    # Test loader doesn't need DistributedSampler
    # (evaluation is typically done on one GPU)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )

    return train_loader, test_loader


def compute_accuracy(model, dataloader, device):
    """
    Compute classification accuracy over a dataset.

    Args:
        model: The neural network model
        dataloader: DataLoader for the dataset
        device: GPU device to run computations on

    Returns:
        accuracy: Float between 0 and 1
    """
    model.eval()
    correct = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()


def main(rank, world_size, num_epochs):
    """
    Main training function for DDP.

    Args:
        rank: GPU index for current process
        world_size: Total number of GPUs
        num_epochs: Number of training epochs
    """
    # Initialize distributed training
    ddp_setup(rank, world_size)

    # Prepare data and model
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)  # Move model to the specific GPU

    # Wrap model with DistributedDataParallel
    # This enables gradient synchronization across GPUs
    model = DDP(model, device_ids=[rank])
    # Note: the core model is now accessible as model.module

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # Training loop
    for epoch in range(num_epochs):
        # Set sampler to ensure different shuffle order each epoch
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:
            # Move data to GPU
            features, labels = features.to(rank), labels.to(rank)

            # Forward pass
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()  # Gradients are automatically synchronized across GPUs
            optimizer.step()

            # Logging (each GPU prints its own progress)
            print(
                f"[GPU{rank}] Epoch: {epoch + 1:03d}/{num_epochs:03d} "
                f"| Batch size: {labels.shape[0]:03d} "
                f"| Loss: {loss:.4f}"
            )

    # Evaluation
    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        test_acc = compute_accuracy(model, test_loader, device=rank)

        # Only print from rank 0 to avoid duplicate outputs
        if rank == 0:
            print("\nTraining completed!")
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")

    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\n"
            f"This script is designed for multiple GPUs. You can run it as:\n"
            f"torchrun --nproc_per_node=2 ddp_training.py\n\n"
            f"Available GPUs: {torch.cuda.device_count()}\n"
            f"If you have {torch.cuda.device_count()} GPUs, uncomment the data "
            f"augmentation lines in prepare_dataset() function."
        ) from e

    # Clean up distributed training
    destroy_process_group()


if __name__ == "__main__":
    # Use environment variables set by torchrun if available
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    # Print system information (only from rank 0 to avoid duplicates)
    if rank == 0:
        print("=" * 60)
        print("PyTorch Distributed Data Parallel (DDP) Training")
        print("=" * 60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"World size: {world_size}")
        print("=" * 60)

        if not torch.cuda.is_available():
            print("\nWARNING: CUDA is not available.")
            print("This script is designed for GPU training.")
            print("Consider using the regular training script for CPU-only systems.")
        elif torch.cuda.device_count() < 2:
            print(f"\nNOTE: Only {torch.cuda.device_count()} GPU detected.")
            print("DDP is most beneficial with 2+ GPUs.")

    # Set random seed for reproducibility
    torch.manual_seed(123)

    # Run training
    num_epochs = 3
    main(rank, world_size, num_epochs)
