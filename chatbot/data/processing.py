from __future__ import annotations

from collections import Counter
from typing import Callable, List

from tqdm import tqdm

from chatbot import logger
from chatbot.data.utils import load_txt_file, normalize_text


class Seq2SeqProcessor:
    """Basic processing for a corpus consisting of pairs of
    text sequences.

    Implements three processing steps:

    .. code-block:: text

        - Normalizes the text sequences
        - Filters the corpus based on the minimum and maximum
        number of allowed tokens in a text sequence
        - Filters the corpus based on the minimum token count
        threshold

    Args:
        pairs (List[List[str]]): The query and response sequence
            pairs.
        tokenizer (Callable[[str], List[str]]): A callable function
            responsible for tokenizing the text sequences.

    Attributes:
        pairs (List[List[str]]): The query and response sequence
            pairs.
        _tokenizer (Callable[[str], List[str]]): A callable function
            responsible for tokenizing the text sequences.

    Example:
        >>> from chatbot.data.processing import Seq2SeqProcessor
        >>> from chatbot.data.tokenizers import get_tokenizer
        >>>
        >>> tokenizer = get_tokenizer()
        >>> processor = Seq2SeqProcessor.from_txt_file(
        ...     file="path-to-parsed-conversations-txt-file",
        ...     tokenizer=tokenizer
        ... )
        >>> filtered_pairs = processor.filter_pairs_by_length(
        ...     pairs=processor.pairs,
        ...     min_length=1,
        ...     max_length=10,
        ... )
        >>> filtered_pairs = processor.filter_pairs_by_token_count(
        ...     pairs=filtered_pairs, min_count=3
        ... )
    """

    def __init__(
        self, pairs: List[List[str]], tokenizer: Callable[[str], List[str]]
    ) -> None:
        self.pairs = pairs
        self._tokenizer = tokenizer

    @classmethod
    def from_txt_file(
        cls,
        file: str,
        tokenizer: Callable[[str], List[str]],
        delimiter: str = "\t",
    ) -> Seq2SeqProcessor:
        """Initializes the `Seq2SeqProcessor` from a `txt` file.

        Normalizes each pair of sequences before initialization.
        Refer to the :func:`chatbot.utils.normalize_text` function
        for what the normalization includes.

        Args:
            file (str): The path to the `txt` file.
            tokenizer (Callable[[str], List[str]]): A callable function
                responsible for tokenizing the text sequences.
            delimiter (str, optional): The string to separate the query
                sequence from the response sequence. Defaults to ``'\t'``.
        """
        lines = load_txt_file(filepath=file)

        pairs = [
            [normalize_text(text) for text in line.split(delimiter)]
            for line in tqdm(lines, desc="Loading corpus...", colour="green")
        ]

        logger.info(f"Loaded {len(pairs)} sequence pairs.")
        return cls(pairs=pairs, tokenizer=tokenizer)

    def _filter_pair_by_length(
        self, pair: List[str], min_length: int, max_length: int
    ) -> bool:
        """Returns ``True`` if both sequences in the provided pair have length
        less than or equal to the maximum threshold.

        Note:
            The sequence length is essentially the length of the
            list of tokens obtained by tokenizing the input sequence.

        Args:
            pair (List[str]): A pair of query and response sequences.
            min_length (int): Minimum number of tokens allowed in each
                of the pair of sequences.
            max_length (int): Maximum number of tokens allowed in each
                of the pair of sequences.
        """
        return (
            min_length <= len(self._tokenizer(pair[0])) <= max_length
            and min_length <= len(self._tokenizer(pair[1])) <= max_length
        )

    def filter_pairs_by_length(
        self, pairs: List[List[str]], min_length: int, max_length: int
    ) -> List[List[str]]:
        """Removing pairs whose sequences (either of them) have length greater
        than or equal to the specified maximum threshold..

        Note:
            The sequence length is essentially the length of the
            list of tokens obtained by tokenizing the input sequence.

        Args:
            pairs (List[List[str]]): The query and response
                sequence pairs.
            min_length (int): Minimum number of tokens allowed in each
                of the pair of sequences.
            max_length (int): Maximum number of tokens allowed in each
                of the pair of sequences.
        Returns:
            List[List[str]]: The filtered pairs.
        """
        return [
            pair
            for pair in pairs
            if self._filter_pair_by_length(pair, min_length, max_length)
        ]

    def filter_pairs_by_token_count(
        self, pairs: List[List[str]], min_count: int
    ) -> List[List[str]]:
        """Removes pairs whose sequences (either of them) include tokens with
        count value less than or equal to the specified minimum threshold.

        Args:
            pairs (List[List[str]]): The query and response
                sequence pairs.
            min_count (int): Minimum token count threshold.

        Returns:
            List[List[str]]: The filtered pairs.
        """
        keep_pairs = []
        token_count: Counter = Counter()
        for pair in pairs:
            for sequence in pair:
                for token in self._tokenizer(sequence):
                    token_count[token] += 1

        for pair in pairs:
            query_sequence, output_sequence = pair[0], pair[1]
            keep_input, keep_output = True, True
            # Check input sequence
            for token in self._tokenizer(query_sequence):
                if token_count[token] <= min_count:
                    keep_input = False
                    break
            # Check output sequence
            for token in self._tokenizer(output_sequence):
                if token_count[token] <= min_count:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s)
            # in their input or output sequence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        return keep_pairs
