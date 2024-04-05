from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import torch

from chatbot.data.vocabulary import SequenceVocabulary


class SequenceVectorizer:
    """Responsible for implementing token-level based vectorization
    of the provided query and response sequences.

    The tokens can be either words or punctuations.

    Each token in the input sequence is substituted with its
    corresponding index from the dataset's vocabulary.

    Args:
        vocab (SequenceVocabulary): Vocabulary object constructed
            from dataset's collection of tokens.
        tokenizer (Callable[[str], List[str]]): Tokenizer for
            splitting the input sequence into tokens.
        max_query_length (int): The length of the longest query
            sequence in the training dataset.
        max_response_length (int): The length of the longest
            response sequence in the training dataset.
    """

    def __init__(
        self,
        vocab: SequenceVocabulary,
        tokenizer: Callable[[str], List[str]],
        max_query_length: int,
        max_response_length: int,
    ) -> None:
        self.vocab: SequenceVocabulary = vocab
        self.tokenizer: Callable[[str], List[str]] = tokenizer
        self.max_query_length = max_query_length + 1
        self.max_response_length = max_response_length + 1

    def _get_query_indices(self, text: str) -> List[int]:
        """Returns a sequence of integers representing
        the query sequence.

        Args:
            text (str): The query sequence.

        Returns:
            List[int]: The token indices of the query
                sequence.
        """
        indices: List[int] = []
        indices.extend(
            self.vocab.lookup_token(token) for token in self.tokenizer(text)
        )
        indices.append(self.vocab.end_seq_index)
        return indices

    def _get_response_indices(self, text: str) -> List[int]:
        """Returns a sequence of integers representing
        the response sequence.

        Args:
            text (str): The response sequence.

        Returns:
            List[int]: The token indices of the response
                sequence.
        """
        indices: List[int] = []
        indices.extend(
            self.vocab.lookup_token(token) for token in self.tokenizer(text)
        )
        indices.append(self.vocab.end_seq_index)
        return indices

    @staticmethod
    def _vectorize(
        indices: List[int], vector_length: int = -1, mask_index: int = 0
    ) -> torch.Tensor:
        """Converts the provided indices to a zero-padded 1-D tensor.

        Args:
            indices (List[str]): List of integers representing
                a text sequence.
            vector_length (int, optional): Size of the output
                tensor. -1 means the length of the input indices
                (default=-1).
            mask_index (int, optional): The mask index to pad the
                sequences, if needed (default=0).

        Returns: vector
            * **vector**: tensor of shape :math:`(L)`

        where :math:`L` is the :attr:`vector_length`.
        """
        if vector_length < 0:
            vector_length = len(indices)

        vector = torch.zeros(size=(vector_length,), dtype=torch.int64)
        vector[: len(indices)] = torch.tensor(indices, dtype=torch.int64)
        vector[len(indices) :] = mask_index  # noqa: E203
        return vector

    def vectorize_single(
        self, query_seq: str, use_dataset_max_length: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """Converts the input query sequence to a zero-padded 1-D tensor.

        The 1-D tensor is appended with the `<EOS>` token index.

        Args:
            query_seq (str): The input sequence.
            use_dataset_max_length (bool, optional): If ``True`` the
                :attr:`query_max_length` is used as the maximum sequence
                length. Otherwise, the maximum length is equal to the
                number of tokens in the provided sequence. Defaults to
                ``True``.

        Returns:
            Tuple[torch.Tensor, int]: The vectorized input sequence and
                the length of it.
        """
        seq_vector_length = -1
        if use_dataset_max_length:
            seq_vector_length = self.max_query_length

        query_indices = self._get_query_indices(text=query_seq)
        query_vector = self._vectorize(
            indices=query_indices,
            vector_length=seq_vector_length,
            mask_index=self.vocab.mask_index,
        )
        return query_vector, len(query_indices)

    def vectorize(
        self,
        query_seq: str,
        response_seq: str,
        use_dataset_max_length: bool = True,
    ) -> Dict[str, torch.Tensor | int]:
        """Converts the provided query and response sequences to
        zero-padded 1-D tensors.

        The query and response sequences are padded with zeros to
        match the maximum length and they are appended with the
        `<EOS>` token index.

        # TODO -> not sure if the `<SOS>` token should be prepended
        # to the response sequence before training.

        Args:
            query_seq (str): The query sequence.
            response_seq (str): The response sequence.
            use_dataset_max_length (bool, optional): If ``True`` the
                :attr:`query_max_length` and :attr:`response_max_length`
                are used as the maximum sequence lengths. Otherwise, the
                maximum lengths are equal to the number of tokens in the
                provided query and response sequences. Defaults to
                ``True``.

        Returns:
            Dict[str, torch.Tensor | int]: The vectorized query and
                response sequences and their lengths.
        """
        query_vector_length = -1
        response_vector_length = -1
        if use_dataset_max_length:
            query_vector_length = self.max_query_length
            response_vector_length = self.max_response_length

        query_indices = self._get_query_indices(text=query_seq)
        query_vector = self._vectorize(
            indices=query_indices,
            vector_length=query_vector_length,
            mask_index=self.vocab.mask_index,
        )

        response_indices = self._get_response_indices(text=response_seq)
        response_vector = self._vectorize(
            indices=response_indices,
            vector_length=response_vector_length,
            mask_index=self.vocab.mask_index,
        )

        return {
            "query_vector": query_vector,
            "response_vector": response_vector,
            "query_length": len(query_indices),
            "response_length": len(response_indices),
        }

    @classmethod
    def from_serializable(cls, contents: Dict[str, Any]) -> SequenceVectorizer:
        """Initializes the vectorizer from a serializable
        dictionary containing the vocabulary, tokenizer and
        the maximum query and response lengths."""
        return cls(
            vocab=contents["vocabulary"],
            tokenizer=contents["tokenizer"],
            max_query_length=contents["max_query_length"],
            max_response_length=contents["max_response_length"],
        )
