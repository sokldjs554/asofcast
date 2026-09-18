"""Deterministic CPU-friendly regression without hidden test-set selection."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_regressor(model: nn.Module, x: np.ndarray, y: np.ndarray, *, epochs: int,
                    learning_rate: float = .001, batch_size: int = 128,
                    seed: int = 42) -> list[float]:
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if x.ndim < 2 or y.ndim != 1 or len(x) != len(y) or len(x) < 1:
        raise ValueError('nonempty input and one scalar target per example required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('training arrays must be finite')
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('invalid training configuration')
    seed_everything(seed)
    inputs = torch.from_numpy(x.copy())
    targets = torch.from_numpy(y.copy())
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(inputs), generator=generator)
        total = 0.0
        for indices in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs[indices])
            loss = torch.mean((prediction - targets[indices]) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss; no checkpoint is promoted')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += loss.item() * len(indices)
        history.append(total / len(inputs))
    model.eval()
    return history


def predict(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if batch_size < 1 or x.ndim < 2 or not np.isfinite(x).all():
        raise ValueError('invalid prediction input')
    if len(x) == 0:
        return np.empty(0, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        result = np.concatenate([model(torch.from_numpy(x[start:start + batch_size].copy())).cpu().numpy()
                                 for start in range(0, len(x), batch_size)])
    if not np.isfinite(result).all():
        raise FloatingPointError('model returned nonfinite prediction')
    return result
