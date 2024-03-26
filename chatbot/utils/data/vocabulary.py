from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from chatbot import logger
from chatbot.utils.aux import normalize_text


class Vocabulary:
    """A class representing a vocabulary for nlp processing
    tasks.

    Maps tokens to indices, indices to tokens and keeps track
    of the count value of each token in the vocabulary and also
    the total token count.

    Args:
        token_to_idx (Dict[str, int], optional):
            A pre-existing mapping of tokens to indices. Defaults
            to `None`.

    Attributes:
        token_to_idx (Dict[str, int]):
            A dictionary mapping tokens to indices.
        idx_to_token (Dict[int, str]:
            A dictionary mapping indices to tokens.
    """

    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx: Dict[str, int] = token_to_idx
        self.token_to_count: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {
            idx: token for token, idx in self.token_to_idx.items()
        }
        self.num_tokens = len(self.token_to_idx)

    def add_token(self, token: str) -> int:
        """Updates the mapping dictionaries based on the
        provided token.

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
        """Updates the mapping dictionaries based on a
        list of input tokens.

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
        begin_seq_index (int): Index associated with the
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
        self.begin_seq_index = self.add_token(token=begin_seq_token)
        self.end_seq_index = self.add_token(token=end_seq_token)

    def lookup_token(self, token: str) -> int:
        """Retrieves the index associated with the token or
        the `UNK` index if token isn't found.

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


class VocabularyBuilder:
    """Constructs a vocabulary object from a list of query and
    response sequence pairs.

    Uses a tokenizer for splitting the text into tokens in order
    to build the vocabulary. Refer to
    :func:`chatbot.data.utils.get_tokenizer` for the available
    tokenizers.

    The `VocabularyBuilder` can be initialized from a `.txt` file
    in which each line is a tab-separated pair of sequences as it
    is shown below.

    .. code-block:: text

        1. They do to!	They do not!
        2. She okay?	I hope so.
        3. Wow	Let's go.
        4. Well, no...	Then that's all you had to say.
        5. Have fun tonight?	Tons

    Args:
        pairs (List[List[str]]): The query and response
            sequence pairs.
        tokenizer: Optional[str]: Name of a tokenizing
            callable function. If ``None`` a `split()`
            tokenizer is used which splits the sequence
            by whitespace. Defaults to ``None``.

    Attributes:
        pairs (List[List[str]]): The query and response
            sequence pairs.
        tokenizer: Callable[[str], List[str]]: A callable
            tokenizer function.

    Example:
        >>> from chatbot.utils.data.vocabulary import VocabularyBuilder
        >>> # Initialize builder from a txt file
        >>> builder = VocabularyBuilder.from_txt_file(
        ...     file="./data/dialogs.txt"
        ... )
        >>>
        >>> # Create the vocabulary
        >>> vocabulary = builder.build_vocabulary(
        ...     max_length=10, min_count=3
        ... )
    """

    def __init__(
        self, pairs: List[List[str]], tokenizer: Callable[[str], List[str]]
    ) -> None:
        self.pairs: List[List[str]] = pairs
        self.tokenizer: Callable[[str], List[str]] = tokenizer
        self.vocab: SequenceVocabulary = SequenceVocabulary()

    @classmethod
    def from_txt_file(
        cls, file: str, tokenizer: Callable[[str], List[str]]
    ) -> VocabularyBuilder:
        """Initializes the `VocabularyBuilder` from a local
        `.txt` file.

        Normalizes each pair of sequences before initialization.
        Refer to the `chatbot.data.utils.normalize_text()` function
        for what the normalization includes.

        Args:
            file (str): Path to the `.txt` file.
            tokenizer: Optional[str]: Name of a tokenizing
                callable function. If ``None`` a `split()`
                tokenizer is used which splits the sequence
                by whitespace. Defaults to ``None``.

        Returns:
            sequencePairProcessor: An instance of the
                `VocabularyBuilder`.
        """
        with open(file=file, mode="r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        pairs = [
            [normalize_text(text) for text in line.split("\t")]
            for line in tqdm(lines, desc="Loading corpus...", colour="green")
        ]

        logger.info(f"Loaded {len(pairs)} sequence pairs.")
        return cls(pairs=pairs, tokenizer=tokenizer)

    def _filter_pair_by_length(self, pair: List[str], max_length: int) -> bool:
        """Returns ``True`` if both sequences in the provided pair
        have length less than or equal to the maximum threshold.

        Note:
            The sequence length is essentially the length of the
            list of tokens obtained by tokenizing the input sequence.

        Args:
            pair (List[str]): A pair of query and response sequences.
            max_length (int): Maximum number of tokens allowed
                in each of the pair of sequences.
        """
        return (
            len(self.tokenizer(pair[0])) <= max_length
            and len(self.tokenizer(pair[1])) <= max_length
        )

    def filter_pairs_by_length(
        self, pairs: List[List[str]], max_length: int
    ) -> List[List[str]]:
        """Removing pairs whose sequences (either of them) have length
        greater than or equal to the specified maximum threshold..

        Note:
            The sequence length is essentially the length of the
            list of tokens obtained by tokenizing the input sequence.

        Args:
            pairs (List[List[str]]): The query and response
                sequence pairs.
            max_length (int): Maximum number of tokens allowed
                in each of the pair of sequences.
        Returns:
            List[List[str]]: The filtered pairs.
        """
        return [pair for pair in pairs if self._filter_pair_by_length(pair, max_length)]

    def filter_pairs_by_token_count(
        self, pairs: List[List[str]], min_count: int
    ) -> List[List[str]]:
        """Removes pairs whose sequences (either of them) include tokens
        with count value less than or equal to the specified minimum
        threshold.

        Args:
            pairs (List[List[str]]): The query and response
                sequence pairs.
            min_count (int): Minimum token count value threshold.

        Returns:
            List[List[str]]: The filtered pairs.
        """
        keep_pairs = []
        token_count: Counter = Counter()
        for pair in pairs:
            for sequence in pair:
                for token in self.tokenizer(sequence):
                    token_count[token] += 1

        for pair in pairs:
            input_sequence, output_sequence = pair[0], pair[1]
            keep_input, keep_output = True, True
            # Check input sequence
            for token in self.tokenizer(input_sequence):
                if token_count[token] <= min_count:
                    keep_input = False
                    break
            # Check output sequence
            for token in self.tokenizer(output_sequence):
                if token_count[token] <= min_count:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s)
            # in their input or output sequence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        logger.info(
            f"Filtered (based on token count) to " f"{len(keep_pairs)} sequence pairs."
        )

        return keep_pairs

    def build_vocabulary(
        self,
        max_length: int = 10,
        min_count: int = 3,
    ) -> Tuple[List[List[str]], SequenceVocabulary]:
        """Filters the sequence pairs and builds the
        vocabulary.

        The filtering includes:

        .. code-block:: text

            - Removing pairs whose sequences (either of them) include tokens
            with count value less than the specified minimum threshold.
            - Removing pairs whose sequences (either of them) have length
            greater than the specified maximum threshold.

        Args:
            max_length (int, optional): Maximum number of tokens allowed
                in each of the pair of sequences (default=10).
            min_count (int, optional): Minimum token count value threshold
                (default=3).

        Returns:
            Tuple[List[List[str]], SequenceVocabulary]: The list of query and
                response sequence pairs and the populated vocabulary.
        """
        filtered_pairs = self.filter_pairs_by_length(
            pairs=self.pairs, max_length=max_length
        )
        logger.info(
            f"Filtered (based on sequence length) to "
            f"{len(filtered_pairs)} sequence pairs."
        )

        filtered_pairs = self.filter_pairs_by_token_count(
            pairs=filtered_pairs, min_count=min_count
        )

        for pair in tqdm(filtered_pairs, desc="Building vocabulary...", colour="green"):
            for sequence in pair:
                self.vocab.add_many(tokens=self.tokenizer(sequence))

        logger.info(
            f"Vocab size: {len(self.vocab)}, "
            f"sequence pairs: {len(filtered_pairs)}\n"
        )
        return filtered_pairs, self.vocab
