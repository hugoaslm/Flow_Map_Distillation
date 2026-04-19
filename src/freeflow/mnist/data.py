from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .config import MnistConfig


class MnistDataModule:
    def __init__(self, config: MnistConfig, device: torch.device, root: str = "./data") -> None:
        self.config = config
        self.device = device
        transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x * 2 - 1)])
        dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        self.loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )
        self.iterator = iter(self.loader)

    def sample_real_images(self, batch_size: int | None = None) -> torch.Tensor:
        if batch_size is not None and batch_size != self.config.batch_size:
            batches = []
            while len(batches) * self.config.batch_size < batch_size:
                batches.append(self.sample_real_images())
            return torch.cat(batches, dim=0)[:batch_size]

        try:
            x, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            x, _ = next(self.iterator)
        return x.to(self.device)

