from typing import Any, Dict

import torch

from chatbot import logger

from .decoders import (
    DecodersFactory,
    LuongAttnDecoderGRU,
    LuongAttnDecoderLSTM,
)
from .encoders import EncoderGRU, EncoderLSTM, EncodersFactory
from .samplers import GreedySearchSampler, RandomSearchSampler, SamplersFactory

AVAILABLE_ENCODERS = ["gru", "lstm"]
AVAILABLE_DECODERS = ["luong_attention_with_gru", "luong_attention_with_lstm"]
AVAILABLE_SAMPLERS = ["greedy", "random"]


def encoders_client(name: str, params: Dict[str, Any]) -> torch.nn.Module:
    """Selects and returns an instance of the selected encoder.

    Args:
        name (str): The name of the encoder to initialize.
        params (Dict[str, Any]): Initialization parameters for
            the selected encoder.

    Returns:
        torch.nn.Module: An instance of the selected encoder.

    Raises:
        ValueError: If the provided encoder name does not exist.
    """
    factory = EncodersFactory()
    match name.lower():
        case "gru":
            factory.register_encoder(name, EncoderGRU)
        case "lstm":
            factory.register_encoder(name, EncoderLSTM)
        case _:
            logger.error(f"Unsupported encoder name: `{name}`")
            raise ValueError(
                f"Unsupported encoder name: `{name}`. "
                f"Choose one of the following: "
                f"{AVAILABLE_ENCODERS}"
            )
    encoder = factory.get_encoder(name, **params)
    return encoder


def decoders_client(name: str, params: Dict[str, Any]) -> torch.nn.Module:
    """Selects and returns an instance of the selected decoder.

    Args:
        name (str): The name of the decoder to initialize.
        params (Dict[str, Any]): Initialization parameters for
            the selected decoder.

    Returns:
        torch.nn.Module: An instance of the selected decoder.

    Raises:
        ValueError: If the provided decoder name does not exist.
    """
    factory = DecodersFactory()
    match name.lower():
        case "luong_attention_with_gru":
            factory.register_decoder(name, LuongAttnDecoderGRU)
        case "luong_attention_with_lstm":
            factory.register_decoder(name, LuongAttnDecoderLSTM)
        case _:
            logger.error(f"Unsupported decoder name: `{name}`")
            raise ValueError(
                f"Unsupported decoder name: `{name}`. "
                f"Choose one of the following: "
                f"{AVAILABLE_DECODERS}"
            )
    decoder = factory.get_decoder(name, **params)
    return decoder


def sampler_client(name: str, params: Dict[str, Any]) -> torch.nn.Module:
    """Selects and returns an instance of the selected sampler.

    Args:
        name (str): The name of the sampler class to initialize.
        params (Dict[str, Any]): Initialization parameters for
            the selected sampler.

    Returns:
        torch.nn.Module: An instance of the selected sampler.

    Raises:
        ValueError: If the provided sampler name does not exist.
    """
    factory = SamplersFactory()
    match name.lower():
        case "greedy":
            factory.register_sampler(name, GreedySearchSampler)
        case "random":
            factory.register_sampler(name, RandomSearchSampler)
        case _:
            logger.error(f"Unsupported sampler name: `{name}`")
            raise ValueError(
                f"Unsupported sampler name: `{name}`. "
                f"Choose one of the following: "
                f"{AVAILABLE_SAMPLERS}"
            )
    sampler = factory.get_sampler(name, **params)
    return sampler
