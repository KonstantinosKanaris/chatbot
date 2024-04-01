import errno
import os
import re
import string
import unicodedata
from typing import Any, Callable, List, Optional

import yaml

from chatbot import logger


def load_yaml_file(filepath: str) -> Any:
    """Loads a `yaml` configuration file into a dictionary.

    Args:
        filepath (str): The path to the `yaml` file.

    Returns:
        Any: The configuration parameters.
    """
    if not os.path.isfile(path=filepath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filepath
        )

    with open(file=filepath, mode="r", encoding="utf-8") as f:
        config = yaml.full_load(stream=f)
    logger.info("Configuration file loaded successfully.\n")

    return config


def _split_tokenizer(text: str) -> List[str]:
    """Splits the input text on whitespace.

    Args:
        text (str): The input text to tokenize.

    Returns:
        List[str]: List with individual text tokens.
    """
    return text.split(" ")


def get_tokenizer(
    tokenizer: Optional[str] = None,
) -> Callable[[str], List[str]]:
    """Returns a tokenizer function for tokenizing
    a string sequence.

    Args:
        tokenizer (str, optional): Name of tokenizer
            function. If ``None`` returns a ``split()``
            function, which splits the string sequence
            by whitespace.

    Returns:
        Callable[[str], List[str]]: The tokenizer function.

    Example:
        >>> from chatbot.utils.aux import get_tokenizer
        >>> split_tokenizer = get_tokenizer()
        >>> tokens = split_tokenizer("A chatbot implementation with PyTorch.")
        >>> tokens
        ['A', 'chatbot', 'implementation', 'with', 'PyTorch.']

    """
    return _split_tokenizer


def _unicode_to_ascii(text: str) -> str:
    """Turns a unicode string to plain `ASCII` character
    by character.

    Args:
        text (str): Input string.

    Returns:
        str: The input text in `ASCII` format.
    """
    all_letters = string.ascii_letters + " .,;'?!"
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


def normalize_text(text: str) -> str:
    """Basic normalization for the provided text.

    Normalization includes:

    .. code-block:: text

        - lowercasing
        - conversion of a unicode string to plain `ASCII`
        - adding whitespace around punctuation symbols
        - trimming all non-letter characters except for basic
        punctuation
        - Replace multiple periods with a single period
        - replace multiple spaces with single space

    Args:
        text (str): Input text for normalizing.

    Returns:
        str: The normalized text.
    """
    text = _unicode_to_ascii(text.lower().strip())
    text = re.sub(pattern=r"([.!?])", repl=r" \1", string=text)
    text = re.sub(pattern=r"[^a-zA-Z.!?]+", repl=r" ", string=text)
    text = re.sub(pattern=r"( . . .)", repl=".", string=text)
    text = re.sub(pattern=r"\s+", repl=" ", string=text).strip()
    return text
