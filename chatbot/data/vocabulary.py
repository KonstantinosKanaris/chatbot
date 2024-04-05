from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from chatbot import logger


class Vocabulary:
    """Creates a vocabulary object which maps tokens to indices.

    Provides methods for adding new tokens and for searching existing
    tokens based on their indices or indices based on their tokens.
    Also keeps track of the count value of each token in the vocabulary.

    Args:
        token_to_idx (Dict[str, int], optional): A pre-existing mapping
            of tokens to indices. Defaults to `None`.

    Attributes:
        token_to_idx (Dict[str, int]):
            A dictionary mapping tokens to indices.
        idx_to_token (Dict[int, str]:
            A dictionary mapping indices to tokens.
    """

    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
        token_to_count: Optional[Dict[str, int]] = None,
    ) -> None:
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = {} if token_to_idx is None else token_to_idx
        self.token_to_count = {} if token_to_count is None else token_to_count
        self.idx_to_token: Dict[int, str] = {
            idx: token for token, idx in self.token_to_idx.items()
        }
        self.num_tokens = len(self.token_to_idx)

    def add_token(self, token: str) -> int:
        """Updates the mapping dictionaries based on the provided token.

        Args:
            token (str): The token to be added to the
                vocabulary.

        Returns:
            int: The index corresponding to the token.
        """
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
            self.token_to_count[token] += 1
        else:
            index = self.num_tokens
            self.token_to_idx[token] = index
            self.token_to_count[token] = 1
            self.idx_to_token[index] = token
            self.num_tokens += 1
        return index

    def add_many(self, tokens: List[str]) -> List[int]:
        """Updates the mapping dictionaries based on a list of input tokens.

        Args:
            tokens (List[str]): A list of string tokens.

        Returns:
            List[int]: A list of indices corresponding
                to the tokens.
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token: str) -> int:
        """Retrieves the index associated with the token.

        Args:
            token (str): The token to look up.

        Returns:
            int: The index corresponding to the token.
        """
        return self.token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        """Retrieves the token associated with the index.

        Args:
            index (int): The index to look up.

        Returns:
            str: The token corresponding to the index.

        Raises:
            KeyError: If the index is not in the vocabulary.
        """
        if index not in self.idx_to_token:
            raise KeyError(f"The index {index} is not in the vocabulary.")
        return self.idx_to_token[index]

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.token_to_idx)

    def __str__(self) -> str:
        """Returns a string representation of the `Vocabulary` instance."""
        return f"<Vocabulary(size={len(self)})>"


class SequenceVocabulary(Vocabulary):
    """Bundles four special tokens used for sequence data.

    The special tokens are: the `UNK` token, the `MASK` token,
    the `START-OF-SEQUENCE` token, and the `END-OF-SEQUENCE`
    token where:

    .. code-block:: text

        * `UNK`: The unknown token used for unseen out-of-vocabulary
        input tokens
        * `MASK`: Enables handling variable-length inputs
        * `START-OF-SEQUENCE`: Start of sequence boundary
        * `END-OF-SEQUENCE`: End of sequence boundary

    Args:
        token_to_idx (Dict[str, int], optional):
            A pre-existing map of tokens to indices. Defaults
            to `None`.
        mask_token (str, optional): The representation of the
            `MASK` token. Defaults to `<MASK>`.
        unk_token (str, optional): The representation of the
            `UNK` token. Defaults to `<UNK>`.
        begin_seq_token (str, optional): The representation of
            the `START-OF-SEQUENCE` token. Defaults to `<SOS>`.
        end_seq_token (str, optional): The representation of
            the `END-OF-SEQUENCE` token. Defaults to `<EOS>`.

    Attributes:
        token_to_idx (Dict[str, int]):
            A dictionary mapping tokens to indices.
        idx_to_token (Dict[int, str]:
            A dictionary mapping indices to tokens.
        _unk_token (str): The representation of the `UNK` token.
        _mask_token (str): The representation of the `MASK` token.
        _begin_seq_token (str): The representation of the
            `START-OF-SEQUENCE` token.
        _end_seq_token (str, optional): The representation of the
            `END-OF-SEQUENCE` token.
        unk_index (int): Index associated with the `UNK` token.
        mask_index (int): Index associated with the `MASK` token.
        start_seq_index (int): Index associated with the
            `START-OF-SEQUENCE` token.
        end_seq_index (int): Index associated with the
            `END-OF-SEQUENCE` token.
    """

    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
        unk_token: str = "<UNK>",
        mask_token: str = "<MASK>",
        begin_seq_token: str = "<SOS>",
        end_seq_token: str = "<EOS>",
    ) -> None:
        super().__init__(token_to_idx=token_to_idx)
        self._mask_token: str = mask_token
        self._unk_token: str = unk_token
        self._begin_seq_token: str = begin_seq_token
        self._end_seq_token: str = end_seq_token

        self.mask_index = self.add_token(token=mask_token)
        self.unk_index = self.add_token(token=unk_token)
        self.start_seq_index = self.add_token(token=begin_seq_token)
        self.end_seq_index = self.add_token(token=end_seq_token)

    def lookup_token(self, token: str) -> int:
        """Retrieves the index associated with the token or the `UNK` index if
        token isn't found.

        Args:
            token (str): The token to look up.

        Returns:
            int: The index corresponding to the token.

        Notes:
            `unk_index` needs to be >=0 (having been added into
            the `Vocabulary`) for the `UNK` functionality.
        """
        if self.unk_index >= 0:
            return self.token_to_idx.get(token, self.unk_index)
        else:
            return self.token_to_idx[token]

    def indices_to_tokens(self, input_tensor: torch.Tensor) -> List[List[str]]:
        """Converts a batch of sequences with token indices to
        a human-readable format.

        Utilizes the vocabulary to match indices to tokens.

        Args:
            input_tensor (torch.Tensor): tensor of shape
                :math:`(N, L)` where :math:`N` the batch
                size and :math:`L` the max sequence length.

        Returns:
            List[List[str]]: List of lists where each inner
                each contains a string with the joined tokens.
        """
        all_token_sequences = []
        for sequence_indices in input_tensor:
            tokens_per_sequence = []
            for token_index in sequence_indices:
                if token_index.item() not in [
                    self.start_seq_index,
                    self.end_seq_index,
                    self.mask_index,
                ]:
                    tokens_per_sequence.append(
                        self.lookup_index(token_index.item())
                    )
            all_token_sequences.append(tokens_per_sequence)

        return all_token_sequences


def build_vocabulary(
    pairs: List[List[str]], tokenizer: Callable[[str], List[str]]
) -> Tuple[SequenceVocabulary, int, int]:
    """Populates the vocabulary with tokens from the provided
    pairs.

    Also, finds the maximum query and response lengths.

    Args:
        pairs (List[List[str]]): The query and response
            sequence pairs.
        tokenizer (Callable[[str], List[str]]): Tokenizer for
            splitting each sequence into tokens.

    Returns:
        Tuple[SequenceVocabulary, int, int]: The populated vocabulary
            and the maximum query and response lengths that were found
            in the provided pairs.
    """
    vocab = SequenceVocabulary()

    bar = tqdm(pairs, "Building vocabulary...", colour="green")
    max_query_length = 0
    max_response_length = 0
    for pair in bar:
        query, response = pair[0], pair[1]
        query_length = len(tokenizer(query))
        response_length = len(tokenizer(response))

        if query_length > max_query_length:
            max_query_length = query_length

        if response_length > max_response_length:
            max_response_length = response_length

        for sequence in pair:
            vocab.add_many(tokens=tokenizer(sequence))

    logger.info(
        f"Vocab size: {len(vocab)}, "
        f"max query length: {max_query_length} "
        f"max response length: {max_response_length}\n"
    )
    return vocab, max_query_length, max_response_length
