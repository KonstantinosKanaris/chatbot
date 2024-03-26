from __future__ import annotations

import os
import time
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import torch

from chatbot import logger


def set_seeds(seed: int = 42) -> None:
    """Sets random seeds for torch operations.

    Args:
      seed (int, optional): Random seed to set (default=42).
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)

    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def load_general_checkpoint(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    filepath: str,
) -> Dict[str, Any]:
    """Loads a general checkpoint.

    Args:
        encoder (torch.nn.Module): The encoder to be updated with its
            saved `state_dict`.
        decoder (torch.nn.Module): The decoder to be updated with its
            saved `state_dict`.
        encoder_optimizer (torch.optim.Optimizer): The optimizer of the
            encoder to be updated with its saved `state_dict`.
        decoder_optimizer (torch.optim.Optimizer): The optimizer of the
            decoder to be updated with its saved `state_dict`.
        filepath (str): The file path of the general checkpoint.
    """
    checkpoint = torch.load(f=filepath)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
    decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
    return {
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"],
    }


class Timer:
    """Context manager to count elapsed time.

    Example:

        >>> def do_something():
        ...     pass
        >>>
        >>> with Timer() as t:
        ...   do_something()
        >>> print(f"Invocation of f took {t.elapsed}s!")
    """

    def __enter__(self) -> Timer:
        """
        Starts the time counting.

        Returns:
          Timer: An instance of the `Timer` class.
        """
        self._start = time.time()
        return self

    def __exit__(self, *args: int | str) -> None:
        """
        Stops the time counting.

        Args:
          args (int | str)
        """
        self._end = time.time()
        self._elapsed = self._end - self._start
        self.elapsed = str(timedelta(seconds=self._elapsed))


class EarlyStopping:
    """Implements early stopping during training.

    Args:
        patience (int, optional):
            Number of epochs to wait before early stopping.
            (default=5).
        delta (float, optional):
            Minimum change in monitored quantity to qualify
            as an improvement (default=0).
        verbose (bool, optional):
            If ``True``, prints a message for each improvement.
            Defaults to `False`.
        path (str, optional):
            Path to save the checkpoint. Should include either
            `.pth` or `.pt` as the file extension. Defaults to
            ``'./checkpoint.pt'``.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        verbose: bool = False,
        path: str = "./checkpoint.pt",
    ) -> None:
        assert os.path.basename(path).endswith(
            (".pth", ".pt")
        ), "model_name should end with '.pt' or '.pth'"

        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose
        self.path: str = path

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min = np.Inf

    def __call__(
        self,
        epoch: int,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        val_loss: float,
    ) -> None:
        """Call method to check if the model's performance has
        improved.

        Args:
            epoch (int): Current epoch.
            encoder (torch.nn.Module): The encoder to be saved.
            decoder (torch.nn.Module): The decoder to be saved.
            encoder_optimizer (torch.optim.Optimizer): The optimizer
                of the encoder.
            decoder_optimizer (torch.optim.Optimizer): The optimizer
                of the decoder.
            val_loss (float): Validation loss to be monitored.
        """
        score = -val_loss

        if not self.best_score:
            self.best_score = score
            self.save_general_checkpoint(
                epoch=epoch,
                encoder=encoder,
                decoder=decoder,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                val_loss=val_loss,
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_general_checkpoint(
                epoch=epoch,
                encoder=encoder,
                decoder=decoder,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                val_loss=val_loss,
            )
            self.counter = 0

    def save_general_checkpoint(
        self,
        epoch: int,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        val_loss: float,
    ) -> None:
        """Saves a general checkpoint during training.

        In addition to the model's `state_dict`, a general checkpoint
        also includes the optimizer's `state_dict`, the current epoch,
        and the validation loss value.

        Args:
            epoch (int): Current epoch.
            encoder (torch.nn.Module): The encoder to be saved.
            decoder (torch.nn.Module): The decoder to be saved.
            encoder_optimizer (torch.optim.Optimizer): The optimizer
                of the encoder.
            decoder_optimizer (torch.optim.Optimizer): The optimizer
                of the decoder.
            val_loss (float): Validation loss at the time of saving
                the checkpoint.
        """
        if not os.path.isdir(s=os.path.dirname(self.path)):
            os.makedirs(name=os.path.dirname(self.path), exist_ok=True)

        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}). Saving model to {self.path}"
            )

        torch.save(
            obj={
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "val_loss": val_loss,
            },
            f=self.path,
        )
        self.val_loss_min = val_loss
