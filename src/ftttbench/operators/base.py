"""Autograd-capable forward operators for DIP, INR and DPS.

The nonlinear Eikonal operator differentiates through its current discrete
linearization. The NumPy Jacobian protocol lives in numpy_base.py.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class ForwardOperator(ABC):
    """Abstract forward model A: x -> y, linear or non-linear.

    Attributes
    ----------
    domain_shape : tuple
        Per-sample shape of x (no batch dim), e.g. ``(C, H, W)``.
    range_shape : tuple
        Per-sample shape of y.
    """

    domain_shape: Tuple[int, ...]
    range_shape: Tuple[int, ...]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A to a batched x. Must support autograd through x."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
