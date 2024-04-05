import codecs
import csv
import errno
import os
import re
import string
import unicodedata
from typing import Any, List

import yaml

from chatbot import logger


def save_pairs(
    pairs: List[List[str]],
    path: str,
    delimiter: str = "\t",
) -> None:
    """Writes the query and response sequence pairs in a `txt` file.

    Each line of the file contains a tab-separated query and response
    sequence pair.

    Args:
        pairs (List[List[str]]): The query and response sequence
            pairs.
        path (str): Path to write the `txt` file.
        delimiter (str, optional): The string to separate the query
            sequence from the response sequence. Defaults to ``'\t'``.
    """
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(name=os.path.dirname(path), exist_ok=True)

    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    logger.info(f"Saving query-response pairs to {path}")
    with open(file=path, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter, lineterminator="\n")
        for pair in pairs:
            writer.writerow(pair)


def load_pairs(filepath: str, delimiter: str = "\t") -> List[List[str]]:
    """Creates a list of lists where each sublist contains
    a query and response sequence pair.

    The pairs are loaded from a `txt` file.

    Args:
        filepath (str): The path to the `txt` file.
        delimiter (str, optional): The string that  separates
            the query sequence from the  response sequence.
            Defaults to ``'\t'``.

    Returns:
        List[List[str]]: The query and response sequence pairs.
    """
    pairs = load_txt_file(filepath=filepath)
    list_of_lists = [
        [text for text in pair.split(delimiter)] for pair in pairs
    ]
    return list_of_lists


def load_yaml_file(filepath: str) -> Any:
    """Loads a `yaml` configuration file into a dictionary.

    Args:
        filepath (str): The path to the `yaml` file.

    Returns:
        Any: The configuration parameters.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    if not os.path.isfile(path=filepath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filepath
        )

    with open(file=filepath, mode="r", encoding="utf-8") as f:
        config = yaml.full_load(stream=f)
    logger.info("Configuration file loaded successfully.\n")

    return config


def load_txt_file(filepath: str) -> List[str]:
    """Creates a list containing the lines
    of a `txt` file.

    Args:
        filepath (str): The path to the `txt` file.

    Returns:
        List[str]: The content of the `txt` file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    if not os.path.isfile(path=filepath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filepath
        )

    with open(file=filepath, mode="r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    return lines


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


def expand_contractions(text: str) -> str:
    """Remove contractions from the input text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without contractions.
    """
    text = text.replace("wouldn't", "would not")
    text = text.replace("shouldn't", "should not")
    text = text.replace("couldn't", "could not")
    text = text.replace("doesn't", "does not")
    text = text.replace("wasn't", "was not")
    text = text.replace("didn't", "did not")
    text = text.replace("don't", "do not")
    text = text.replace("won't", "will not")
    text = text.replace("can't", "cannot")
    text = text.replace("haven't", "have not")
    text = text.replace("hadn't", "had not")
    text = text.replace("weren't", "were not")
    text = text.replace("aren't", "are not")
    text = text.replace("isn't", "is not")
    text = text.replace("hasn't", "has not")
    text = text.replace("mustn't", "must not")

    text = text.replace("where'd", "where did")
    text = text.replace("why'd", "why did")
    text = text.replace("how'd", "how did")

    text = text.replace("doin'", "doing")
    text = text.replace("'til", "until")
    text = text.replace("goin'", "going")
    text = text.replace("ma'am", "madam")
    text = text.replace("'bout", "about")
    text = text.replace("'em", "them")

    text = text.replace("s'posed", "supposed")
    text = text.replace("d'ya", "do you")
    text = text.replace("c'mon", "common")
    text = text.replace("let's", "let us")

    # more general contractions
    text = re.sub(pattern=r"\'m", repl=" am", string=text)
    text = re.sub(pattern=r"\'re", repl=" are", string=text)
    text = re.sub(pattern=r"\'ve", repl=" have", string=text)
    text = re.sub(pattern=r"\'s", repl=" is", string=text)
    text = re.sub(pattern=r"\'ll", repl=" will", string=text)
    return text


def normalize_text(text: str) -> str:
    """Basic normalization for the provided text.

    Normalization includes:

    .. code-block:: text

        - Lowercasing
        - Conversion of a unicode string to plain `ASCII`
        - Expanding contractions, .e.g, `'I'm'` -> `'I am'`
        - Trimming all non-letter characters except for basic
        punctuation
        - Replacing consecutive punctuation with a single
        punctuation, e.g., `'???'` -> `'?'`
        - Adding whitespace around punctuation symbols
        - Replace multiple spaces with single space

    Args:
        text (str): Input text for normalizing.

    Returns:
        str: The normalized text.
    """
    text = _unicode_to_ascii(text.lower().strip())
    text = expand_contractions(text=text)
    text = re.sub(pattern=r"[^a-zA-Z.!?]+", repl=r" ", string=text)
    text = re.sub(pattern=r"([.!?,;])\1+", repl=r" \1 ", string=text)
    text = re.sub(pattern=r"\s+", repl=" ", string=text).strip()
    text = re.sub(pattern=r"(\. ){1,}\.", repl=r".", string=text)
    text = re.sub(pattern=r"([.!?])", repl=r" \1 ", string=text)
    text = (
        re.sub(pattern=r"\s+", repl=" ", string=text)
        .strip()
        .lstrip(".")
        .strip()
    )
    return text
