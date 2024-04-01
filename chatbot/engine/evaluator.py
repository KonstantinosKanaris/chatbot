from typing import List

import torch

from chatbot import logger
from chatbot.utils.aux import normalize_text
from chatbot.utils.data.vectorizer import SequenceVectorizer


class Evaluator:
    def __init__(
        self,
        vectorizer: SequenceVectorizer,
    ) -> None:
        self.vectorizer = vectorizer
        self.vocab = vectorizer.vocab
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def evaluate(self, input_sequence: str, searcher: torch.nn.Module) -> List[str]:
        """Generates a list of tokens in response to the input query.

        Implements a forward pass through the decoder and samples its
        predictions to obtain the best response tokens at each time-step.

        Args:
            input_sequence (str): The input query.
            searcher (torch.nn.Module): A search decoder for sampling the
                decoder's predictions. Available searcher decoders:
                `GreedySearchDecoder` and `RandomSearchDecoder`.

        Returns:
            List[str]: The generated tokens.
        """
        token_indices, indices_length, _ = self.vectorizer.vectorize(
            sequence=input_sequence, use_dataset_max_length=True
        )
        token_indices = token_indices.unsqueeze(dim=1).to(self.device)
        indices_length_tensor = torch.tensor(
            data=indices_length, dtype=torch.int64, device=torch.device("cpu")
        ).unsqueeze(0)

        predicted_token_indices = searcher(
            token_indices,
            indices_length_tensor,
            self.vectorizer.max_seq_length,
        )
        decoded_words = [
            self.vectorizer.vocab.lookup_index(idx.item())
            for idx in predicted_token_indices
        ]

        decoded_words[:] = [
            token
            for token in decoded_words
            if token
            not in [
                self.vocab.lookup_index(self.vocab.end_seq_index),
                self.vocab.lookup_index(self.vocab.start_seq_index),
                self.vocab.lookup_index(self.vocab.mask_index),
            ]
        ]
        return decoded_words

    def chat(self, searcher: torch.nn.Module) -> None:
        """Starts an interactive conversation with the bot."""
        while True:
            try:
                input_sequence = input("> ")
                if input_sequence in ["q", "quit", "quit()", ""]:
                    break
                normalized_seq = normalize_text(text=input_sequence)
                output_tokens = self.evaluate(
                    input_sequence=normalized_seq, searcher=searcher
                )
                print(f"Bot: {' '.join(output_tokens)}")
            except KeyError:
                logger.exception("Error: Encountered unknown word")
