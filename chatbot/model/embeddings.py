from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from torch import nn
from tqdm.auto import tqdm


class PreTrainedEmbeddings:
    """A wrapper around pretrained word vectors and their use.

    Loads the pretrained embeddings from a file and creates an
    embedding matrix ready to be passed into an `nn.Embedding` layer.

    Args:
        word_to_index (Dict[str, int]): A mapping from words
            to indexes.
        word_vectors (List[torch.Tensors]): List with word
            embeddings.

    Attributes:
        word_to_index (Dict[str, int]): A mapping from words
            to indexes.
        word_vectors (List[torch.Tensors]): List with word
            embeddings.
        index_to_word (Dict[int, str]): A mapping from indexes
            to words.

    Example:

        >>> # Example loading glove embeddings
        >>> embeddings_wrapper = PreTrainedEmbeddings.from_embeddings_file(
        ...     embedding_file="./.vector_cache/glove.6B.100d.txt"
        ... )
        >>> word_vectors = embeddings_wrapper.word_vectors
        >>> word_to_index = embeddings_wrapper.word_to_index
        >>> index_to_word = embeddings_wrapper.index_to_word
        >>> print(len(word_vectors))
        400000
        >>> print(word_vectors[0].shape)
        torch.Size([100])

        >>> # Create an embedding matrix for a given list of words
        >>> embeddings_matrix = embeddings_wrapper.make_embedding_matrix(
        ...     words=["Creating", "a", "matrix"]
        ... )
        >>> print(embeddings_matrix.shape)
        torch.Size([3, 100])

        >>> # Get embedding for a single word
        >>> cat_embedding = embeddings_wrapper.get_embedding(word="cat")
        >>> print(cat_embedding)
        tensor([0.2309, 0.2828, 0.6318, 0.5941, 0.5860, 0.6326, 0.2440, 0.1411,
         0.0608, -0.7898, -0.2910,  0.1429,  0.7227,  0.2043,  0.1407,  0.9876,
         0.5253,  0.0975,  0.8822,  0.5122,  0.4020,  0.2117, -0.0131, -0.7162,
         0.5539,  1.1452, -0.8804, -0.5022, -0.2281,  0.0239,  0.1072,  0.0837,
         0.5501,  0.5848,  0.7582,  0.4571, -0.2800,  0.2522,  0.6896, -0.6097,
         0.1958,  0.0442, -0.3114, -0.6883, -0.2272,  0.4618, -0.7716,  0.1021,
         0.5564,  0.0674, -0.5721,  0.2374,  0.4717,  0.8277, -0.2926, -1.3422,
        -0.0993,  0.2814,  0.4160,  0.1058,  0.6220,  0.8950, -0.2345,  0.5135,
         0.9938,  1.1846, -0.1636,  0.2065,  0.7385,  0.2406, -0.9647,  0.1348,
        -0.0072,  0.3302, -0.1236,  0.2719, -0.4095,  0.0219, -0.6069,  0.4076,
         0.1957, -0.4180,  0.1864, -0.0327, -0.7857, -0.1385,  0.0440, -0.0844,
         0.0491,  0.2410,  0.4527, -0.1868,  0.4618,  0.0891, -0.1819, -0.0152,
        -0.7368, -0.1453,  0.1510, -0.7149])
    """

    def __init__(
        self, word_to_index: Dict[str, int], word_vectors: List[torch.Tensor]
    ) -> None:
        self.word_to_index: Dict[str, int] = word_to_index
        self.word_vectors: List[torch.Tensor] = word_vectors
        self.index_to_word: Dict[int, str] = {
            index: word for word, index in word_to_index.items()
        }

    @classmethod
    def from_embeddings_file(cls, embedding_file: str) -> PreTrainedEmbeddings:
        """Instantiates the wrapper from a pre-trained vector file.

        Vector file should have the following format:

        .. code-block:: text

            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N

        Args:
            embedding_file (str): The path to the embedding file.

        Returns:
            PreTrainedEmbeddings:
                An instance of the `PretrainedEmbeddings` class.
        """
        word_to_index: Dict[str, int] = {}
        word_vectors = []
        with open(file=embedding_file, mode="r", encoding="utf-8") as f:
            tqdm_bar = tqdm(
                iterable=f.readlines(),
                position=0,
                leave=True,
                desc="Reading word vectors ...",
                colour="green",
            )
            for line in tqdm_bar:
                line = line.split(" ")
                word = line[0]
                vector = torch.tensor(
                    data=[float(x) for x in line[1:]], dtype=torch.float32
                )

                word_to_index[word] = len(word_to_index)
                word_vectors.append(vector)

        return cls(word_to_index, word_vectors)

    def make_embedding_matrix(self, words: List[str]) -> torch.Tensor:
        r"""Creates an embedding matrix for a specific set of words.

        Args:
            words (List[str]): A list of words.

        Returns: final_embeddings
            * **final_embeddings**: tensor of shape :math:`(V, D)`.

            where:

            .. math::
                \begin{aligned}
                    N ={} & \text{vocabulary size} \\
                    D ={} & \text{dimension of token embeddings}
                \end{aligned}
        """
        embeddings = torch.stack(self.word_vectors)
        embedding_size = embeddings.size(1)

        final_embeddings = torch.zeros(
            size=(len(words), embedding_size), dtype=torch.float32
        )

        for i, word in enumerate(words):
            if word in self.word_to_index:
                final_embeddings[i, :] = embeddings[self.word_to_index[word]]
            else:
                embedding_i = torch.ones(size=(1, embedding_size))
                torch.nn.init.xavier_uniform_(embedding_i)
                final_embeddings[i, :] = embedding_i

        return final_embeddings

    def get_embedding(self, word: str) -> torch.Tensor:
        """Returns the embedding of the input word.

        Args:
            word (str): The input word to be embedded.

        Returns:
            torch.Tensor: The word embedding.
        """
        return self.word_vectors[self.word_to_index[word]]

    def __len__(self) -> int:
        """Returns the size of the pretrained embeddings."""
        return len(self.word_to_index)


class EmbeddingLayerConstructor:
    """Responsible for initializing an `nn.Embedding` layer.

    Initializes the embedding layer with either pretrained
    frozen weights from a file or with learnable weights.

    Example:
        >>> from chatbot.models.embeddings import EmbeddingLayerConstructor
        >>> embedding_init_params = {"embedding_dim": 100, "padding_idx": 0}
        >>> embedding = EmbeddingLayerConstructor.create_embedding_layer(
        ...     vocab_tokens=["1st_token", "2nd_token", "and_so_on"],
        ...     filepath="./pretrained_embeddings/glove.6B.100d.txt",
        ...     **embedding_init_params
        ...     )
        >>>
        >>> embedding
    """

    @classmethod
    def from_file(
        cls,
        vocab_tokens: List[str],
        filepath: str,
        **kwargs,
    ) -> nn.Embedding:
        """Loads pretrained word embeddings, i.e., GLoVe embeddings,
        into a `nn.Embedding` layer.

        Args:
            vocab_tokens (List[str]): All the tokens in the
                vocabulary.
            filepath (str): Path to the pretrained embeddings file.
            **kwargs: Initialization parameters of the embedding layer,
                e.g., `embedding_dim`.

        Returns:
            nn.Embedding: An embedding layer with pretrained frozen weights.
        """
        embeddings_wrapper = PreTrainedEmbeddings.from_embeddings_file(
            embedding_file=filepath
        )
        input_embeddings = embeddings_wrapper.make_embedding_matrix(
            words=vocab_tokens
        )
        embedding = nn.Embedding(
            num_embeddings=len(vocab_tokens), **kwargs
        ).from_pretrained(embeddings=input_embeddings, freeze=True)
        return embedding

    @classmethod
    def without_pretrained_embeddings(
        cls, vocab_tokens: List[str], **kwargs
    ) -> nn.Embedding:
        """Returns an initialized `nn.Embedding` layer with learnable
        weights.

        Args:
            vocab_tokens (List[str]): All the tokens in the
                vocabulary.
            **kwargs: Initialization parameters of the embedding layer,
                e.g., `embedding_dim`.

        Returns:
            nn.Embedding: An embedding layer with learnable weights.
        """
        return nn.Embedding(num_embeddings=len(vocab_tokens), **kwargs)

    @classmethod
    def create_embedding_layer(
        cls,
        vocab_tokens: List[str],
        filepath: Optional[str] = None,
        **kwargs,
    ) -> nn.Embedding:
        """Initializes a `nn.Embedding` layer.

        If :attr:`filepath` is provided the embedding layer is
        initialized with pretrained word embeddings from the file.
        Otherwise, returns an embedding layer with learnable weights.

        Args:
            vocab_tokens (List[str]): All the tokens in the vocabulary.
            filepath (str, optional): Path to the pretrained embeddings
                file. Defaults to ``None``.
            **kwargs: Initialization parameters of the embedding layer,
                e.g., `embedding_dim`.

        Returns:
            nn.Embedding: An embedding layer with either pretrained
                frozen weights or with learnable weights.
        """
        if filepath and os.path.exists(filepath):
            embedding = cls.from_file(vocab_tokens, filepath, **kwargs)
        else:
            embedding = cls.without_pretrained_embeddings(
                vocab_tokens, **kwargs
            )
        return embedding
