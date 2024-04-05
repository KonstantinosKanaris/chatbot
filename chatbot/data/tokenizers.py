from typing import Callable, Dict, List, Optional


class TokenizersFactory:
    """Responsible for registering and creating different
    text tokenizers.

    Each tokenizer should be a callable function that takes
    as input a string and returns a list of strings.

    Attributes:
        _tokenizers (Dict[str, Callable[[str], List[str]]]):
            A dictionary to store registered tokenizers.
    """

    def __init__(self) -> None:
        self._tokenizers: Dict[str, Callable[[str], List[str]]] = {}

    def add_tokenizer(
        self, name: str, tokenizer: Callable[[str], List[str]]
    ) -> None:
        """Registers a new tokenizer to the factory.

        Args:
            name (str): The name of the tokenizer.
            tokenizer (Callable[[str], List[str]]): The tokenizer
                method.
        """
        self._tokenizers[name] = tokenizer

    def get_tokenizer(self, name: str) -> Callable[[str], List[str]]:
        """Returns the selected tokenizer by name.

        Args:
            name (str): The name of the tokenizer to get.

        Returns:
            Callable[[str], List[str]]: The tokenizer.
        """
        tokenizer = self._tokenizers[name]
        return tokenizer


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
        >>> split_tokenizer = get_tokenizer()
        >>> tokens = split_tokenizer("A chatbot implementation with PyTorch.")
        >>> tokens
        ['A', 'chatbot', 'implementation', 'with', 'PyTorch.']

    """
    return _split_tokenizer
