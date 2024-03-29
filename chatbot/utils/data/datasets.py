from __future__ import annotations

from typing import Callable, Dict, List

import torch
from torch.utils.data import Dataset

from chatbot.utils.data.vectorizer import SequenceVectorizer
from chatbot.utils.data.vocabulary import VocabularyBuilder


class CornellDialogsDataset(Dataset):
    def __init__(
        self, sequence_pairs: List[List[str]], vectorizer: SequenceVectorizer
    ) -> None:
        self.sequence_pairs: List[List[str]] = sequence_pairs
        self._vectorizer: SequenceVectorizer = vectorizer

    @classmethod
    def load_pairs_and_vectorizer(
        cls,
        file: str,
        tokenizer: Callable[[str], List[str]],
        max_length: int = 10,
        min_count: int = 3,
    ) -> CornellDialogsDataset:
        """Initializes the `CornelDialogsDataset` from a `.txt` file
        that contains the sequence pairs.

        Args:
            file (str): Path to the `.txt` file.
            tokenizer (Callable[[str], List[str]]: Tokenizer for
                splitting each sequence before vectorization.
            max_length (int, optional): Maximum number of tokens allowed
                in each of the pair sequence (default=10).
            min_count (int, optional): Minimum token count value threshold
                (default=3).

        Returns:
            CornelDialogsDataset: An instance of `CornelDialogsDataset`.
        """
        vocab_builder = VocabularyBuilder.from_txt_file(file=file, tokenizer=tokenizer)

        pairs, vocabulary = vocab_builder.build_vocabulary(
            max_length=max_length,
            min_count=min_count,
        )

        return cls(
            sequence_pairs=pairs,
            vectorizer=SequenceVectorizer.from_serializable(
                contents={
                    "vocabulary": vocabulary,
                    "tokenizer": tokenizer,
                    "max_seq_len": max_length,
                }
            ),
        )

    def get_vectorizer(self) -> SequenceVectorizer:
        """Returns the vectorizer."""
        return self._vectorizer

    def get_num_batches(self, batch_size: int) -> int:
        """Given a batch size, returns the number of batches
        in the dataset.

        Args:
            batch_size (int): The batch size.

        Returns:
            int: The number of batches in the dataset.
        """
        return len(self) // batch_size

    def __len__(self) -> int:
        """Returns the number of the total sequence pairs."""
        return len(self.sequence_pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        r"""Returns a data point in vectorized form based on the input index.

        Refer to `chatbot.utils.data..vectorizer.SequenceVectorizer`.

        Args:
            index (int): The index to the data point.

        Returns:
            Dict[str, torch.Tensor | int]: A dictionary containing:

            * **input_sequence**: tensor of shape :math:`(L)`
            * **target_sequence**: tensor of shape :math:`(L)`
            * **target_mask**: tensor of shape :math:`(L)`
            * **input_length**: The length of the input sequence
            * **target_length**: The length of the target sequence

            where:

            .. math::
                \begin{aligned}
                    L ={} & \text{sequence length} \\
                \end{aligned}
        """
        sequence_pair = self.sequence_pairs[index]
        query_sequence = sequence_pair[0]
        response_sequence = sequence_pair[1]

        input_seq, input_length, _ = self._vectorizer.vectorize(
            sequence=query_sequence, use_dataset_max_length=True
        )
        target_seq, target_length, target_mask = self._vectorizer.vectorize(
            sequence=response_sequence, use_dataset_max_length=True
        )

        return {
            "input_sequence": input_seq,
            "target_sequence": target_seq,
            "target_mask": target_mask,
            "input_length": input_length,
            "target_length": target_length,
        }
