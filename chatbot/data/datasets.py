from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from chatbot.data.vectorizer import SequenceVectorizer


class CornellDialogsDataset(Dataset):
    def __init__(
        self, pairs: List[List[str]], vectorizer: SequenceVectorizer
    ) -> None:
        self.pairs: List[List[str]] = pairs
        self._vectorizer: SequenceVectorizer = vectorizer

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
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        r"""Returns a data point in vectorized form based on the
        input index.

        Refer to `chatbot.data.vectorizer.SequenceVectorizer`.

        Args:
            index (int): The index to the data point.

        Returns:
            Dict[str, torch.Tensor | int]: A dictionary containing:

            * **x_query**: tensor of shape :math:`(L_{query})`.
            * **y_response**: tensor of shape :math:`(L_{response})`.
            * **x_query_length**: The length of the query sequence.
            * **y_response_length**: The length of the response sequence.

            where:

            .. math::
                \begin{aligned}
                    L_{query} ={} & \text{query sequence length} \\
                    L_{response} ={} & \text{response sequence length} \\
                \end{aligned}
        """
        sequence_pair = self.pairs[index]
        query_sequence = sequence_pair[0]
        response_sequence = sequence_pair[1]

        vector_dict = self._vectorizer.vectorize(
            query_seq=query_sequence,
            response_seq=response_sequence,
            use_dataset_max_length=True,
        )

        return {
            "x_query": vector_dict["query_vector"],
            "y_response": vector_dict["response_vector"],
            "query_length": vector_dict["query_length"],
            "response_length": vector_dict["response_length"],
        }
