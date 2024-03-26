from typing import Tuple

import torch
from torch import nn

from .attention import AttnLayer


class LuongAttnDecoderRNN(nn.Module):
    r"""Decoder layer for a seq2seq model.

    Produces a probability distribution predicting the next
    token in a sequence of tokens.

    The decoder uses a global attention mechanism as described
    on the paper "Effective Approaches to Attention-based Neural
    Machine Translation". Luong, M.-T., Pham, H., & Manning, C. D.
    (2015). https://arxiv.org/abs/1508.04025

    Args:
        alignment_method (str): Name of the alignment score method used
            in the attention layer. The available methods are: `concat`,
            `dot` and `general`.
            (vocabulary size).
        embedding (nn.Embedding): The embedding layer.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The size of the vocabulary.
        num_layers (int, optional): The number of recurrent layers (default=1).
        dropout (float, optional) The dropout value applied to the embedded
            input sequence and to the GRU layer. If non-zero, the :attr:`num_layers`
            should be greater than 1, otherwise the dropout value in the GRU layer
            will be set to zero. (default=0.1).

    Inputs: input_seq, h_0, encoder_state
        * **input_seq**: tensor of shape :math:`(1, N)`.
        * **h_0**: tensor of shape :math:`(num_layers, N, H)`.
        * **encoder_state**: tensor of shape :math:`(L, N, H)`

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{max sequence length} \\
                H ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, hidden
        * **output**: tensor of shape :math:`(N, V)`.
        * **hidden**: tensor of shape :math:`(num_layers, N, H)`.

        where:

        .. math::
            \begin{aligned}
                V ={} & \text{vocabulary size} \\
            \end{aligned}
    """

    def __init__(
        self,
        alignment_method: str,
        embedding: nn.Embedding,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if alignment_method not in ["concat", "dot", "general"]:
            raise ValueError(f"'{alignment_method}' is not a valid attention method.")

        self.num_layers = num_layers

        self.emb = embedding

        self.emb_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
        )
        self.concat = nn.Linear(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attention = AttnLayer(method=alignment_method, hidden_size=hidden_size)

    def forward(
        self,
        input_seq: torch.Tensor,
        h_0: torch.Tensor,
        encoder_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the decoder.

        Note:
            The decoder runs one step (token) at a time.

        Args:
            input_seq (torch.Tensor): Batch of single token sequences.
            h_0 (torch.Tensor): The last hidden state of the encoder.
            encoder_state (torch.Tensor): The output of the encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predictive distribution
                and the attentional hidden state of the decoder.
        """
        embedded_input = self.emb_dropout(self.emb(input_seq))

        rnn_output, hidden = self.gru(embedded_input, h_0)
        attn_energies = self.attention(rnn_output, encoder_state)

        # Multiply attention weights to encoder outputs to get a new
        # 'weighted sum' context vector
        context = attn_energies.bmm(encoder_state.transpose(0, 1))

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(dim=0)
        context = context.squeeze(dim=1)
        concat_input = torch.cat((rnn_output, context), dim=1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict new token using Luong eq. 6
        output = torch.softmax(self.out(concat_output), dim=1)

        return output, hidden
