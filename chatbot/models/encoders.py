from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    r"""Encoder layer for a seq2seq model.

    Encodes a variable-length input sequence of tokens into a
    fixed-length vector representation.

    Args:
        embedding (nn.Embedding): The embedding layer.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int, optional): The number of recurrent layers (default=1).
        dropout (float, optional): The dropout value applied to the GRU layer.
            If non-zero, the :attr:`num_layers` should be greater than 1,
            otherwise the dropout value will be set to zero. (default=0).

    Inputs: input_sequences, input_lengths, h_0
        * **input_sequences**: tensor of shape :math:`(L, N)`.
        * **input_lengths**: tensor of shape :math:`(N)`.
        * **h_0**: tensor of shape :math:`(2 * \text{num_layers}, H)` or
          :math:`(2 * \text{num_layers}, N, H)` containing the initial
          hidden state for the input sequence. Defaults to ``None``.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{max sequence length} \\
                H ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, N, H)`.
        * **h_n**: tensor of shape :math:`(2 * \text{num_layers}, N, H)`.

    Attributes:
        hidden_size (int): The number of features in the hidden state `h`.
        emb (torch.nn.Module): The embedding layer for the input sequence.
        bi_gru (torch.nn.Module): A bidirectional GRU layer.

    Example:
        >>> num_vocab_tokens = 1000
        >>> batch_size = 10
        >>> max_seq_len = 12
        >>>
        >>> # Create embedding layer
        >>> embedding = nn.Embedding(
        ...     embedding_dim=100,
        ...     num_embeddings=num_vocab_tokens,
        ...     padding_idx=0
        ... )
        >>> # Create encoder layer
        >>> encoder = EncoderRNN(
        ...     embedding=embedding,
        ...     hidden_size=100,
        ...     num_layers=1,
        ...     dropout=0
        ... )
        >>> # Create a sample of input sequences. In practice sequences
        >>> # with length less than the maximum are padded with the mask index
        >>> input_sequences = torch.randint(
        ...     low=0,
        ...     high=num_vocab_tokens,
        ...     size=(batch_size, max_seq_len),
        ...     dtype=torch.int64
        ... )
        >>> # Find the length of each sequence (without the mask index)
        >>> # in the input sequences
        >>> input_lengths = torch.sum(input_sequences != 0, dim=1)

        >>> output, h_n = encoder(
        ...     input_sequences=input_sequences,
        ...     input_lengths=input_lengths
        ... )
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size

        self.emb = embedding

        self.bi_gru = nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            bidirectional=True,
        )

    def forward(
        self,
        input_sequences: torch.Tensor,
        input_lengths: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the encoder.

        Args:
            input_sequences (torch.Tensor): A batch of input sequences.
            input_lengths (torch.Tensor): Holds the length of each
                input sequence.
            h_0 (torch.Tensor, optional): Initial hidden state
                of the rnn layer of the encoder. Defaults to
                ``None``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output and hidden
                state of the encoder layer.
        """
        embedded_input = self.emb(input_sequences)

        packed_input = pack_padded_sequence(
            input=embedded_input,
            lengths=input_lengths,
        )

        output, h_n = self.bi_gru(packed_input, h_0)

        output, _ = pad_packed_sequence(sequence=output)

        # Sum bidirectional GRU outputs
        output = (
            output[:, :, : self.hidden_size]
            + output[:, :, self.hidden_size :]  # noqa: E203
        )

        return output, h_n
