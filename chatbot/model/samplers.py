from typing import Dict

import torch
from torch import nn


class SamplersFactory:
    """A factory class for registering and instantiating
    decoder samplers.

    Attributes:
        _samplers (Dict[str, Any]): A dictionary to store
            registered models.
    """

    def __init__(self) -> None:
        self._samplers: Dict[str, type[torch.nn.Module]] = {}

    def register_sampler(
        self, name: str, sampler: type[torch.nn.Module]
    ) -> None:
        """Registers a sample with the given name.

        Args:
            name (str): The name of the model to register.
            sampler (type[torch.nn.Module]): The sampler's class
                type.
        """
        self._samplers[name] = sampler

    def get_sampler(self, name: str, **kwargs) -> torch.nn.Module:
        """Instantiates and returns a sampler by name.

        Args:
            name (str): The name of the sampler to instantiate.
            **kwargs: Initialization parameters for the sampler.

        Returns:
            torch.nn.Module: An instance of the specified sampler.
        """
        sampler = self._samplers[name]
        return sampler(**kwargs)


class GreedySearchSampler(nn.Module):
    r"""Implements greedy search decoding.

    Selects the most probable token, i.e `argmax` from the model's
    vocabulary at each decoding time-step as candidate to the output
    sequence.

    Args:
        encoder (torch.nn.Module): The encoder layer of a seq-to-seq model.
        decoder (torch.nn.Module): The decoder layer of a seq-to-seq model.
        eos_idx (int): The end-of-sentence token index.

    Inputs: input_seq, query_length, max_seq_length
        * **input_seq**: tensor of shape :math:`(L_{in}, 1)`.
        * **query_length**: tensor of shape :math:`(1, )`.
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
        query_length: torch.Tensor,
        max_seq_length: int,
    ) -> torch.Tensor:
        """Forward pass of the Greedy Search Sampler.

        Args:
            input_seq (torch.Tensor): Input tokenized sequence.
            query_length (torch.Tensor): Length of input sequence.
            max_seq_length (int): The maximum sequence length in the
                entire dataset.

        Returns:
            torch.Tensor: The predicted output indices.
        """
        self.encoder.eval()
        self.decoder.eval()

        with torch.inference_mode():

            encoder_cell = None
            if self.encoder.__class__.__name__ == "EncoderGRU":
                encoder_state, encoder_hidden = self.encoder(
                    input_seq, query_length
                )
            else:
                encoder_state, encoder_hidden, encoder_cell = self.encoder(
                    input_seq, query_length
                )

            decoder_hidden = encoder_hidden[: self.decoder.num_layers]
            decoder_cell = None
            if (
                self.decoder.__class__.__name__ == "LuongAttnDecoderLSTM"
                and isinstance(encoder_cell, torch.Tensor)
            ):
                decoder_cell = encoder_cell[: self.decoder.num_layers]

            decoder_input = torch.ones(
                size=(1, 1), device=self.device, dtype=torch.int64
            )

            # Initialize tensors to append decoded indices to
            all_token_indices = torch.zeros(
                size=[0], device=self.device, dtype=torch.int64
            )

            for _ in range(max_seq_length):
                if self.decoder.__class__.__name__ == "LuongAttnDecoderGRU":
                    decoder_output, decoder_hidden = self.decoder(
                        input_seq=decoder_input,
                        h_0=decoder_hidden,
                        encoder_state=encoder_state,
                    )
                else:
                    decoder_output, decoder_hidden, decoder_cell = (
                        self.decoder(
                            input_seq=decoder_input,
                            h_0=decoder_hidden,
                            c_0=decoder_cell,
                            encoder_state=encoder_state,
                        )
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
    r"""Implements multinomial sampling search during decoding.

    Samples a token according to the output probability distribution over
    the vocabulary.

    Args:
        encoder (torch.nn.Module): The encoder layer of a seq-to-seq model.
        decoder (torch.nn.Module): The decoder layer of a seq-to-seq model.
        eos_idx (int): The end-of-sentence token index.

    Inputs: input_seq, query_length, max_seq_length
        * **input_seq**: tensor of shape :math:`(L_{in}, 1)`.
        * **query_length**: tensor of shape :math:`(1, )`.
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
        query_length: torch.Tensor,
        max_seq_length: int,
    ) -> torch.Tensor:
        """Forward pass of the Greedy Search Sampler.

        Args:
            input_seq (torch.Tensor): Input tokenized sequence.
            query_length (torch.Tensor): Length of input sequence.
            max_seq_length (int): The maximum sequence length in the
                entire dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted token indices
                along with their predicted scores.
        """
        self.encoder.eval()
        self.decoder.eval()

        with torch.inference_mode():
            # Encoder Forward pass
            encoder_cell = None
            if self.encoder.__class__.__name__ == "EncoderGRU":
                encoder_state, encoder_hidden = self.encoder(
                    input_seq, query_length
                )
            else:
                encoder_state, encoder_hidden, encoder_cell = self.encoder(
                    input_seq, query_length
                )

            decoder_hidden = encoder_hidden[: self.decoder.num_layers]
            decoder_cell = None
            if (
                self.decoder.__class__.__name__ == "LuongAttnDecoderLSTM"
                and isinstance(encoder_cell, torch.Tensor)
            ):
                decoder_cell = encoder_cell[: self.decoder.num_layers]

            decoder_input = torch.ones(
                size=(1, 1), device=self.device, dtype=torch.int64
            )

            # Initialize tensors to append decoded indices to
            all_token_indices = torch.zeros(
                size=[0], device=self.device, dtype=torch.int64
            )

            for _ in range(max_seq_length):
                if self.decoder.__class__.__name__ == "LuongAttnDecoderGRU":
                    decoder_output, decoder_hidden = self.decoder(
                        input_seq=decoder_input,
                        h_0=decoder_hidden,
                        encoder_state=encoder_state,
                    )
                else:
                    decoder_output, decoder_hidden, decoder_cell = (
                        self.decoder(
                            input_seq=decoder_input,
                            h_0=decoder_hidden,
                            c_0=decoder_cell,
                            encoder_state=encoder_state,
                        )
                    )

                decoder_input = torch.multinomial(
                    input=decoder_output, num_samples=1
                ).squeeze(dim=0)

                all_token_indices = torch.cat(
                    (all_token_indices, decoder_input), dim=0
                )

                if decoder_input.item() == self.eos_idx:
                    break
                else:
                    decoder_input = decoder_input.unsqueeze(dim=0)

        return all_token_indices
