from typing import Callable

import torch
from torch import nn


class AttnLayer(nn.Module):
    r"""An alignment attention layer for producing attention
    energies.

    The implementation is based on the paper "Effective Approaches
    to Attention-based Neural Machine Translation". Luong, M.-T., Pham,
    H., & Manning, C. D. (2015). https://arxiv.org/abs/1508.04025

    Applies an alignment score method for computing the attention energies.
    The available alignment score methods are: ``concat``, ``dot`` and
    ``general``.

    Args:
        hidden_size (int): Hidden size of `Linear` layer
            used in `concat` and `general` alignment function.
        method (str, optional): The name of the alignment function.
            Defaults to `concat`.

    Inputs: decoder_output, encoder_state
        * **decoder_output**: tensor of shape :math:`(1, N, H)`.
        * **encoder_state**: tensor of shape :math:`(L, N, H)`

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{max sequence length} \\
                H ={} & \text{hidden_size}
            \end{aligned}

    Outputs: attention_energies
        * **attention_energies**: tensor of shape :math:`(N, 1, L)`
    """

    def __init__(
        self,
        hidden_size: int,
        method: str = "concat",
    ) -> None:
        super().__init__()
        if method not in ["concat", "dot", "general"]:
            raise ValueError(f"'{method}' is not a valid attention method.")
        self.method: str = method

        if self.method == "concat":
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(data=torch.FloatTensor(1, hidden_size))

        if self.method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)

    def _concat_score(
        self, decoder_output: torch.Tensor, encoder_state: torch.Tensor
    ) -> torch.Tensor:
        """Produces attention energies using the `concat` product
         alignment function.

        Args:
            decoder_output (torch.Tensor): The ith output of the decoder.
            encoder_state (torch.Tensor): The output of the encoder.

        Returns:
            torch.Tensor: The attention energies.
        """
        energy = self.attn(
            torch.cat(tensors=(decoder_output, encoder_state), dim=2)
        ).tanh()
        return torch.sum(input=decoder_output * energy, dim=2)

    @staticmethod
    def _dot_score(
        decoder_output: torch.Tensor, encoder_state: torch.Tensor
    ) -> torch.Tensor:
        """Produces attention energies using the `dot` product
         alignment function.

        Args:
            decoder_output (torch.Tensor): The ith output of the decoder.
            encoder_state (torch.Tensor): The output of the encoder.

        Returns:
            torch.Tensor: The attention energies.
        """
        return torch.sum(input=decoder_output * encoder_state, dim=2)

    def _general_score(
        self,
        decoder_output: torch.Tensor,
        encoder_state: torch.Tensor,
    ) -> torch.Tensor:
        """Produces attention energies using the `general` alignment
        function.

        Args:
            decoder_output (torch.Tensor): The ith output of the decoder.
            encoder_state (torch.Tensor): The output of the encoder.

        Returns:
            torch.Tensor: The attention energies.
        """
        energy = self.attn(encoder_state)
        return torch.sum(input=(decoder_output * energy), dim=2)

    def select_attention_score(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns an alignment function for producing attention
        energies.

        Available alignment functions: `concat`, `dot`, and `general`.

        Returns:
            A `Callable` alignment function.
        """
        if self.method == "concat":
            return self._concat_score
        elif self.method == "dot":
            return self._dot_score
        else:
            return self._general_score

    def forward(
        self,
        decoder_output: torch.Tensor,
        encoder_state: torch.Tensor,
    ) -> torch.Tensor:
        """The forward pass of the attention layer.

        Args:
            decoder_output (torch.Tensor): The ith output of the decoder.
            encoder_state (torch.Tensor): The output of the encoder.

        Returns:
            torch.Tensor: The attention energies (weights).
        """
        score_method = self.select_attention_score()
        attn_energies = score_method(decoder_output, encoder_state)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        return torch.softmax(input=attn_energies, dim=1).unsqueeze(dim=1)
