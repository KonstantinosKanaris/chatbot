from typing import Dict, Tuple

import torch
from torch import nn

from .attention import AttnLayer


class DecodersFactory:
    """A factory class for registering and instantiating
    decoders.

    Attributes:
        _decoders (Dict[str, Any]): A dictionary to store
            registered decoders.
    """

    def __init__(self) -> None:
        self._decoders: Dict[str, type[torch.nn.Module]] = {}

    def register_decoder(
        self, name: str, decoder: type[torch.nn.Module]
    ) -> None:
        """Registers a decoder with the given name.

        Args:
            name (str): The name of the decoder to register.
            decoder (type[torch.nn.Module]): The decoder's class
                type.
        """
        self._decoders[name] = decoder

    def get_decoder(self, name: str, **kwargs) -> torch.nn.Module:
        """Instantiates and returns a decoder by name.

        Args:
            name (str): The name of the decoder to instantiate.
            **kwargs: Initialization parameters for the decoder.

        Returns:
            torch.nn.Module: An instance of the specified decoder.
        """
        decoder = self._decoders[name]
        return decoder(**kwargs)


class LuongAttnDecoderGRU(nn.Module):
    r"""Decoder with GRU layer for a sequence-to-sequence model.

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
        num_layers (int, optional): The number of recurrent layers
            (default=1).
        dropout (float, optional): The dropout value applied to the
            embedded input sequence and to the GRU layer. If non-zero,
            the :attr:`num_layers` should be greater than 1, otherwise
            the dropout value will be set to zero. (default=0.1).
        temperature (float, optional): Used to control the randomness of
            the decoder's predictions by scaling the logits before applying
            softmax. Must be greater than zero. If ``temperature = 1`` there
            is no effect to the output logits (default=1).

    Inputs: input_seq, h_0, encoder_state
        * **input_seq**: tensor of shape :math:`(1, N)`.
        * **h_0**: tensor of shape :math:`(num_layers, N, H)`.
        * **encoder_state**: tensor of shape :math:`(L, N, H)`.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{max sequence length} \\
                H ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(N, V)`.
        * **h_n**: tensor of shape :math:`(num_layers, N, H)`.

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
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if alignment_method not in ["concat", "dot", "general"]:
            raise ValueError(
                f"'{alignment_method}' is not a valid attention method."
            )

        self.num_layers = num_layers
        self.emb = embedding
        self.temperature = temperature

        self.emb_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            input_size=self.emb.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
        )
        self.concat = nn.Linear(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attention = AttnLayer(
            method=alignment_method, hidden_size=hidden_size
        )

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

        rnn_output, h_n = self.gru(embedded_input, h_0)
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
        output = torch.softmax(
            self.out(concat_output) * self.temperature, dim=1
        )

        return output, h_n


class LuongAttnDecoderLSTM(nn.Module):
    r"""Decoder with LSTM layer for a sequence-to-sequence model.

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
        num_layers (int, optional): The number of recurrent layers
            (default=1).
        dropout (float, optional): The dropout value applied to the
            embedded input sequence and to the GRU layer. If non-zero,
            the :attr:`num_layers` should be greater than 1, otherwise
            the dropout value will be set to zero. (default=0.1).
        temperature (float, optional): Used to control the randomness of
            the decoder's predictions by scaling the logits before applying
            softmax. Must be greater than zero. If ``temperature = 1`` there
            is no effect to the output logits (default=1).

    Inputs: input_seq, h_0, c_0, encoder_state
        * **input_seq**: tensor of shape :math:`(1, N)`.
        * **h_0**: tensor of shape :math:`(num_layers, N, H)`.
        * **c_0**: tensor of shape :math:`(num_layers, N, H)`.
        * **encoder_state**: tensor of shape :math:`(L, N, H)`.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{max sequence length} \\
                H ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(N, V)`.
        * **h_n**: tensor of shape :math:`(num_layers, N, H)`.

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
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if alignment_method not in ["concat", "dot", "general"]:
            raise ValueError(
                f"'{alignment_method}' is not a valid attention method."
            )

        self.num_layers = num_layers
        self.emb = embedding
        self.temperature = temperature

        self.emb_dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=self.emb.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
        )
        self.concat = nn.Linear(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attention = AttnLayer(
            method=alignment_method, hidden_size=hidden_size
        )

    def forward(
        self,
        input_seq: torch.Tensor,
        h_0: torch.Tensor,
        c_0: torch.Tensor,
        encoder_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The forward pass of the decoder.

        Note:
            The decoder runs one step (token) at a time.

        Args:
            input_seq (torch.Tensor): Batch of single token sequences.
            h_0 (torch.Tensor): The last hidden state of the encoder.
            c_0 (torch.Tensor):
            encoder_state (torch.Tensor): The output of the encoder.
        """
        embedded_input = self.emb_dropout(self.emb(input_seq))

        rnn_output, (h_n, c_n) = self.lstm(embedded_input, (h_0, c_0))
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
        output = torch.softmax(
            self.out(concat_output) * self.temperature, dim=1
        )

        return output, h_n, c_n
