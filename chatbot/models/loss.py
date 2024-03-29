import torch
from torch import nn


class MaskedNLLLoss(nn.Module):
    r"""Computes the negative log likelihood loss.

    Ignores target values, i.e., PAD values, specified by the
    input `mask` tensor.

    Inputs: y_pred, y_true, mask
        * **y_pred**: tensor of shape :math:`(N, V)`.
        * **y_true**: tensor of shape :math:`(N)`.
        * **mask**: tensor of shape :math:`(N)`.

    where:

    .. math::
        \begin{aligned}
            N ={} & \text{batch size} \\
            V ={} & \text{vocabulary size}
        \end{aligned}

    Outputs: loss
        * **loss**: tensor of shape :math:`()`.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(
        y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        cross_entropy = -torch.log(
            torch.gather(input=y_pred, dim=1, index=y_true.view(-1, 1))
        )
        loss = cross_entropy.masked_select(mask).mean()
        return loss
