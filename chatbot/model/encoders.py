from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncodersFactory:
    """A factory class for registering and instantiating
    encoders.

    Attributes:
        _encoders (Dict[str, Any]): A dictionary to store
            registered encoders.
    """

    def __init__(self) -> None:
        self._encoders: Dict[str, type[torch.nn.Module]] = {}

    def register_encoder(
        self, name: str, encoder: type[torch.nn.Module]
    ) -> None:
        """Registers an encoder with the given name.

        Args:
            name (str): The name of the encoder to register.
            encoder (type[torch.nn.Module]): The encoder's class
                type.
        """
        self._encoders[name] = encoder

    def get_encoder(self, name: str, **kwargs) -> torch.nn.Module:
        """Instantiates and returns an encoder by name.

        Args:
            name (str): The name of the encoder to instantiate.
            **kwargs: Initialization parameters for the encoder.

        Returns:
            torch.nn.Module: An instance of the specified encoder.
        """
        encoder = self._encoders[name]
        return encoder(**kwargs)


class EncoderGRU(nn.Module):
    r"""Encoder with a bidirectional GRU layer for a
    sequence-to-sequence model.

    Encodes a batch of variable-length query sequences, comprised
    of tokens, into a fixed-length vector representation.

    Args:
        embedding (nn.Embedding): The embedding layer.
        hidden_size (int): The number of features in  the hidden
            state.
        num_layers (int, optional): The number of GRU layers
            (default=1).
        dropout (float, optional): The dropout value applied to
            the GRU layer. If non-zero,
            the :attr:`num_layers` should be greater
            than 1, otherwise the dropout value will
            be set to zero. (default=0).

    Inputs: x_queries, query_lengths, h_0
        * **x_queries**: tensor of shape :math:`(L, N)`.
        * **query_lengths**: tensor of shape :math:`(N)`.
        * **h_0**: tensor of shape  :math:`(2 * \text{num_layers}, H)`
        or :math:`(2 * \text{num_layers}, N, H)`  containing the
        initial hidden state for  the input sequence. Defaults to ``None``.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{max sequence length} \\
                H ={} & \text{hidden_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, N, H)`.
        * **h_n**: tensor of shape :math:`(2 * \text{num_layers}, N, H)`.

    Attributes:
        hidden_size (int): The number of features in the hidden state `h`.
        emb (torch.nn.Module): The embedding layer for the input sequence.
        bi_gru (torch.nn.Module): A bidirectional GRU layer.
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
        x_queries: torch.Tensor,
        query_lengths: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the encoder.

        Args:
            x_queries (torch.Tensor): A batch of input sequences.
            query_lengths (torch.Tensor): Holds the length of each
                input sequence.
            h_0 (torch.Tensor, optional): Initial hidden state
                of the rnn layer of the encoder. Defaults to
                ``None``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output and hidden
                state of the encoder layer.
        """
        embedded_input = self.emb(x_queries)

        packed_input = pack_padded_sequence(
            input=embedded_input,
            lengths=query_lengths,
        )

        output, h_n = self.bi_gru(packed_input, h_0)

        output, _ = pad_packed_sequence(sequence=output)

        # Sum bidirectional GRU outputs
        output = (
            output[:, :, : self.hidden_size]
            + output[:, :, self.hidden_size :]  # noqa: E203
        )

        return output, h_n


class EncoderLSTM(nn.Module):
    r"""Encoder with a bidirectional LSTM layer for a
    sequence-to-sequence model.

    Encodes a batch of variable-length query sequences, comprised of
    tokens, into a fixed-length vector representation.

    Args:
        embedding (nn.Embedding): The embedding layer.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int, optional): The number of recurrent layers (default=1).
        dropout (float, optional): The dropout value applied to the GRU layer.
            If non-zero, the :attr:`num_layers` should be greater than 1,
            otherwise the dropout value will be set to zero. (default=0).

    Inputs: x_queries, query_lengths, h_0
        * **x_queries**: tensor of shape :math:`(L, N)`.
        * **query_lengths**: tensor of shape :math:`(N)`.
        * **h_0**: tensor of shape :math:`(2 * \text{num_layers}, H)` or
          :math:`(2 * \text{num_layers}, N, H)` containing the initial
          hidden state for the input sequence. Defaults to ``None``.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{max sequence length} \\
                H ={} & \text{hidden_size}
            \end{aligned}

    Outputs: output, h_n, c_n
        * **output**: tensor of shape :math:`(L, N, H)`.
        * **h_n**: tensor of shape :math:`(2 * \text{num_layers}, N, H)`.
        * **c_n**: tensor of shape :math:`(2 * \text{num_layers}, N, H)`.

    Attributes:
        hidden_size (int): The number of features in the hidden state `h`.
        emb (torch.nn.Module): The embedding layer for the input sequence.
        bi_lstm (torch.nn.Module): A bidirectional LSTM layer.
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

        self.bi_lstm = nn.LSTM(
            input_size=self.emb.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            bidirectional=True,
        )

    def forward(
        self,
        x_queries: torch.Tensor,
        query_lengths: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The forward pass of the encoder.

        Args:
            x_queries (torch.Tensor): A batch of input sequences.
            query_lengths (torch.Tensor): Holds the length of each
                input sequence.
            h_0 (torch.Tensor, optional): Initial hidden state
                of the rnn layer of the encoder. Defaults to
                ``None``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output
                ,cell, and hidden state of the encoder layer.
        """
        embedded_input = self.emb(x_queries)

        packed_input = pack_padded_sequence(
            input=embedded_input,
            lengths=query_lengths,
        )

        output, (h_n, c_n) = self.bi_lstm(packed_input, h_0)

        output, _ = pad_packed_sequence(sequence=output)

        # Sum bidirectional GRU outputs
        output = (
            output[:, :, : self.hidden_size]
            + output[:, :, self.hidden_size :]  # noqa: E203
        )

        return output, h_n, c_n
