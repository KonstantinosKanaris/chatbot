from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import torch

from chatbot.utils.data.vocabulary import SequenceVocabulary


class SequenceVectorizer:
    """Responsible for implementing token-level based vectorization
    of an input text sequence.

    The tokens can be either words or punctuations.

    Each token in the input sequence is substituted with each
    corresponding index in the sequence vocabulary.

    Args:
        vocab (SequenceVocabulary): Vocabulary object constructed
            from dataset's collection of tokens.
        tokenizer (Callable[[str], List[str]]): Tokenizer for
            splitting the input sequence into tokens.
        max_seq_length (int): The length of the longest sequence
            in the dataset.
    """

    def __init__(
        self,
        vocab: SequenceVocabulary,
        tokenizer: Callable[[str], List[str]],
        max_seq_length: int,
    ) -> None:
        self.vocab: SequenceVocabulary = vocab
        self.tokenizer: Callable[[str], List[str]] = tokenizer
        # +2 for <SOS>, and <EOS> tokens
        self.max_seq_length = max_seq_length + 2

    def vectorize(
        self, sequence: str, use_dataset_max_length: bool = True
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Converts an input text sequence into a 1-D tensor of
        integers.

        The input sequence is padded with zeros to match the maximum
        length and is appended with the `<EOS>` token.

        Args:
            sequence (str): Input text sequence.
            use_dataset_max_length (bool, optional): If ``True``
                the value of :attr:`max_seq_length` is used as
                the maximum sequence length. Otherwise, the maximum
                length is equal to the number of tokens in the input
                sequence. Defaults to ``True``.

        Returns:
            Tuple[torch.Tensor, int, torch.Tensor]: A tuple containing
                the vectorized sequence, the length of the un-padded
                sequence and the mask of the sequence where mask is the
                boolean version of the vectorized sequence.
        """
        # indices = [self.vocab.start_seq_index]
        indices: List[int] = []
        indices.extend(
            self.vocab.lookup_token(token)
            for token in self.tokenizer(sequence)
        )
        indices.append(self.vocab.end_seq_index)

        if use_dataset_max_length:
            vector_length = self.max_seq_length
        else:
            vector_length = len(indices)

        output = torch.zeros(size=(vector_length,), dtype=torch.int64)
        output[: len(indices)] = torch.tensor(indices, dtype=torch.int64)
        output[len(indices) :] = self.vocab.mask_index  # noqa: E203
        mask = output.bool()
        return output, len(indices), mask

    @classmethod
    def from_serializable(cls, contents: Dict[str, Any]) -> SequenceVectorizer:
        """Initializes the vectorizer from a serializable
        dictionary containing the vocabulary, tokenizer and
        the maximum sequence length."""
        return cls(
            vocab=contents["vocabulary"],
            tokenizer=contents["tokenizer"],
            max_seq_length=contents["max_seq_len"],
        )
