from __future__ import annotations

import math
import os
import time
from datetime import timedelta
from typing import Optional

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


def inverse_sigmoid_decay(decay: int | float, epoch: int) -> float:
    r"""Implements an inverse sigmoid decay scheduler.

    The implementation is based on the paper `Scheduled Sampling for
    Sequence Prediction with Recurrent Neural Networks`. Samy Bengio,
    Oriol Vinyals, Navdeep Jaitly, Noam Shazeer.
    https://arxiv.org/abs/1506.03099.

    Reduces the probability of selecting the actual target token over
    the decoder's prediction to be fed into the decoder at the next
    time step.

    The probability :math:`e_{i}` at epoch :math:`i` and with a
    :attr:`decay` value :math:`k` is computed with the following
    function:

    .. math::
        \begin{aligned}
            e_i = \frac{k}{k + exp(\frac{i}{k})}, \text{ where } k \ge 1
        \end{aligned}

    Args:
          decay (int): The scheduled sampling decay factor.
          epoch (int): Running epoch.

    Returns:
        float: The probability of selecting the actual target as next
            input to the decoder.
    """
    assert decay >= 1, "Decay must be greater than or equal to 1."
    return decay / (decay + math.exp(epoch / decay))


def write_checkpoint(
    epoch: int,
    embedding: torch.nn.Embedding,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    val_loss: float,
    path: str = "./checkpoint.pth",
) -> None:
    """Saves the states of the training components of a seq-to-seq
    model and also the last validation loss and epoch.

    Args:
        epoch (int): Current epoch.
        embedding (torch.nn.Embedding): The embedding layer.
        encoder (torch.nn.Module): The encoder layer.
        decoder (torch.nn.Module): The decoder layer.
        encoder_optimizer (torch.optim.Optimizer): The optimizer of
            the encoder.
        decoder_optimizer (torch.optim.Optimizer): The optimizer of
            the decoder.
        val_loss (float): Loss value during validation.
        path (str, optional): Path to write the checkpoint.
            Should include either `.pth` or `.pt` as the file
            extension. Defaults to ``'./checkpoint.pth'``.
    """
    if not os.path.isdir(s=os.path.dirname(path)):
        os.makedirs(name=os.path.dirname(path), exist_ok=True)

    logger.info(f"Saving checkpoint to {path}\n")

    torch.save(
        obj={
            "epoch": epoch,
            "embedding": embedding.state_dict(),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "encoder_optimizer": encoder_optimizer.state_dict(),
            "decoder_optimizer": decoder_optimizer.state_dict(),
            "val_loss": val_loss,
        },
        f=path,
    )


class EarlyStopping:
    """Stops the training process and checkpoints the states of
    the training components of a seq-to-seq model.

    The stopping occurs if the loss value during the validation
    step stops decreasing for a number of epochs specified by
    the :attr:`patience`.

    Args:
        patience (int, optional): Number of epochs to wait
            before early stopping. (default=5).
        delta (float, optional): Minimum change in monitored
            quantity to qualify as an improvement (default=0).
        verbose (bool, optional): If ``True``, logs a message
            for each improvement. Defaults to `False`.
        path (str, optional): Path to write the checkpoint.
            Should include either `.pth` or `.pt` as the file
            extension. Defaults to ``'./checkpoint.pth'``.
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
        embedding: torch.nn.Embedding,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        val_loss: float,
    ) -> None:
        """Call method to check if the model's performance has improved.

        Checkpointing takes place only for the lowest validation loss
        values during training.

        Args:
            epoch (int): Current epoch.
            embedding (torch.nn.Embedding): The embedding layer.
            encoder (torch.nn.Module): The encoder layer.
            decoder (torch.nn.Module): The decoder layer.
            encoder_optimizer (torch.optim.Optimizer): The optimizer
                of the encoder.
            decoder_optimizer (torch.optim.Optimizer): The optimizer
                of the decoder.
            val_loss (float): Validation loss to be monitored.
        """
        score = -val_loss

        if not self.best_score:
            self.best_score = score

            if self.verbose:
                logger.info(
                    f"Validation loss decreased "
                    f"({self.val_loss_min:.6f} --> {val_loss:.6f})."
                )

            write_checkpoint(
                epoch=epoch,
                embedding=embedding,
                encoder=encoder,
                decoder=decoder,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                val_loss=val_loss,
                path=self.path,
            )

            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score

            if self.verbose:
                logger.info(
                    f"Validation loss decreased "
                    f"({self.val_loss_min:.6f} --> {val_loss:.6f})."
                )

            write_checkpoint(
                epoch=epoch,
                embedding=embedding,
                encoder=encoder,
                decoder=decoder,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                val_loss=val_loss,
                path=self.path,
            )

            self.counter = 0
            self.val_loss_min = val_loss


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
        """Starts the time counting.

        Returns:
          Timer: An instance of the `Timer` class.
        """
        self._start = time.time()
        return self

    def __exit__(self, *args: int | str) -> None:
        """Stops the time counting.

        Args:
          args (int | str)
        """
        self._end = time.time()
        self._elapsed = self._end - self._start
        self.elapsed = str(timedelta(seconds=self._elapsed))
