from typing import List, Tuple

import torch
from torch import nn

from chatbot import logger
from chatbot.utils.aux import normalize_text
from chatbot.utils.data.vectorizer import SequenceVectorizer


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(
        self,
        input_seq: torch.Tensor,
        input_length: torch.Tensor,
        max_seq_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_state, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[: self.decoder.num_layers]
        decoder_input = torch.ones(size=(1, 1), device=self.device, dtype=torch.int64)

        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros(size=[0], device=self.device, dtype=torch.int64)
        all_scores = torch.zeros(size=[0], device=self.device)

        for _ in range(max_seq_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_state
            )
            decoder_score, decoder_input = torch.max(decoder_output, dim=1)

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_score), dim=0)

            decoder_input = decoder_input.unsqueeze(dim=0)

        return all_tokens, all_scores


class Evaluator:
    def __init__(
        self,
        vectorizer: SequenceVectorizer,
    ) -> None:
        self.vectorizer = vectorizer
        self.vocab = vectorizer.vocab
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def evaluate(self, input_sequence: str, searcher: torch.nn.Module) -> List[str]:
        token_indices, indices_length, _ = self.vectorizer.vectorize(
            sequence=input_sequence, use_dataset_max_length=True
        )
        token_indices = token_indices.unsqueeze(dim=1).to(self.device)
        indices_length_tensor = torch.tensor(
            data=indices_length, dtype=torch.int64, device=self.device
        ).unsqueeze(0)

        predicted_token_indices, scores = searcher(
            token_indices, indices_length_tensor, self.vectorizer.max_seq_length
        )
        decoded_words = [
            self.vectorizer.vocab.lookup_index(idx.item())
            for idx in predicted_token_indices
        ]
        return decoded_words

    def chat(self, searcher) -> None:
        while True:
            try:
                input_sequence = input("> ")
                if input_sequence in ["q", "quit", "quit()", ""]:
                    break
                normalized_seq = normalize_text(text=input_sequence)
                output_tokens = self.evaluate(
                    input_sequence=normalized_seq, searcher=searcher
                )
                output_tokens[:] = [
                    token
                    for token in output_tokens
                    if token
                    not in [
                        self.vocab.lookup_index(self.vocab.end_seq_index),
                        self.vocab.lookup_index(self.vocab.begin_seq_index),
                        self.vocab.lookup_index(self.vocab.mask_index),
                    ]
                ]
                print(f"Bot: {' '.join(output_tokens)}")
            except KeyError:
                logger.exception("Error: Encountered unknown word")
