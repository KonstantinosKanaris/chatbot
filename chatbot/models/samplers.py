import torch
from torch import nn


class GreedySearchSampler(nn.Module):
    r"""Implements greedy decoding.

    Selects the most probable token, i.e `argmax` from the model's
    vocabulary at each decoding time-step as candidate to the output
    sequence.

    Args:
        encoder (torch.nn.Module): The encoder layer of a seq-to-seq model.
        decoder (torch.nn.Module): The decoder layer of a seq-to-seq model.
        eos_idx (int): The end-of-sentence token index.

    Inputs: input_seq, input_length, max_seq_length
        * **input_seq**: tensor of shape :math:`(L_{in}, 1)`.
        * **input_length**: tensor of shape :math:`(1, )`.
        * **max_seq_length**: An integer number.

    Outputs: all_token_indices
        * **all_token_indices**: tensor of shape :math:`(L_{out})`.

        where:

        .. math::
            \begin{aligned}
                L_{in} ={} & \text{input sequence length} \\
                L_{out} ={} & \text{max_seq_length}
            \end{aligned}
    """

    def __init__(
        self, encoder: torch.nn.Module, decoder: torch.nn.Module, eos_idx: int
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.eos_idx = eos_idx

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(
        self,
        input_seq: torch.Tensor,
        input_length: torch.Tensor,
        max_seq_length: int,
    ) -> torch.Tensor:
        """Forward pass of the Greedy Search Sampler.

        Args:
            input_seq (torch.Tensor): Input tokenized sequence.
            input_length (torch.Tensor): Length of input sequence.
            max_seq_length (int): The maximum sequence length in the
                entire dataset.

        Returns:
            torch.Tensor: The predicted output indices.
        """
        self.encoder.eval()
        self.decoder.eval()

        with torch.inference_mode():
            encoder_state, encoder_hidden = self.encoder(
                input_seq, input_length
            )
            decoder_hidden = encoder_hidden[: self.decoder.num_layers]
            decoder_input = torch.ones(
                size=(1, 1), device=self.device, dtype=torch.int64
            )

            # Initialize tensors to append decoded indices to
            all_token_indices = torch.zeros(
                size=[0], device=self.device, dtype=torch.int64
            )

            for _ in range(max_seq_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_state
                )
                decoder_input = torch.argmax(decoder_output, dim=1)

                all_token_indices = torch.cat(
                    (all_token_indices, decoder_input), dim=0
                )

                if decoder_input.item() == self.eos_idx:
                    break
                else:
                    decoder_input = decoder_input.unsqueeze(dim=0)

        return all_token_indices


class RandomSearchSampler(nn.Module):
    r"""Implements multinomial sampling during decoding.

    Samples a token according to the output probability distribution over
    the vocabulary.

    Args:
        encoder (torch.nn.Module): The encoder layer of a seq-to-seq model.
        decoder (torch.nn.Module): The decoder layer of a seq-to-seq model.
        eos_idx (int): The end-of-sentence token index.

    Inputs: input_seq, input_length, max_seq_length
        * **input_seq**: tensor of shape :math:`(L_{in}, 1)`.
        * **input_length**: tensor of shape :math:`(1, )`.
        * **max_seq_length**: An integer number.

    Outputs: all_token_indices
        * **all_token_indices**: tensor of shape :math:`(L_{out})`.

        where:

        .. math::
            \begin{aligned}
                L_{in} ={} & \text{input sequence length} \\
                L_{out} ={} & \text{max_seq_length}
            \end{aligned}
    """

    def __init__(
        self, encoder: torch.nn.Module, decoder: torch.nn.Module, eos_idx: int
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.eos_idx = eos_idx

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(
        self,
        input_seq: torch.Tensor,
        input_length: torch.Tensor,
        max_seq_length: int,
    ) -> torch.Tensor:
        """Forward pass of the Greedy Search Sampler.

        Args:
            input_seq (torch.Tensor): Input tokenized sequence.
            input_length (torch.Tensor): Length of input sequence.
            max_seq_length (int): The maximum sequence length in the
                entire dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted token indices
                along with their predicted scores.
        """
        self.encoder.eval()
        self.decoder.eval()

        with torch.inference_mode():
            encoder_state, encoder_hidden = self.encoder(
                input_seq, input_length
            )
            decoder_hidden = encoder_hidden[: self.decoder.num_layers]
            decoder_input = torch.ones(
                size=(1, 1), device=self.device, dtype=torch.int64
            )

            # Initialize tensors to append decoded indices to
            all_token_indices = torch.zeros(
                size=[0], device=self.device, dtype=torch.int64
            )

            for _ in range(max_seq_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_state
                )
                decoder_input = torch.multinomial(
                    input=torch.softmax(decoder_output, dim=1), num_samples=1
                ).squeeze(dim=0)

                all_token_indices = torch.cat(
                    (all_token_indices, decoder_input), dim=0
                )

                if decoder_input.item() == self.eos_idx:
                    break
                else:
                    decoder_input = decoder_input.unsqueeze(dim=0)

        return all_token_indices
