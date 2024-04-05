from typing import Any, Dict

import torch

from chatbot import logger

from .samplers import GreedySearchSampler, RandomSearchSampler, SamplersFactory

AVAILABLE_SAMPLERS = ["greedy", "random"]


def sampler_client(name: str, params: Dict[str, Any]) -> torch.nn.Module:
    """Selects and returns an instance of the selected
    sampler.

    Args:
        name (str): The name of the sampler class to
            initialize.
        params (Dict[str, Any]): Initialization parameters
            of the selected sampler.

    Returns:
        torch.nn.Module: An instance of the selected
            sampler.

    Raises:
        ValueError: If the provided sampler name does
            not exist.
    """
    factory = SamplersFactory()
    match name.lower():
        case "greedy":
            factory.register_model(name, GreedySearchSampler)
        case "random":
            factory.register_model(name, RandomSearchSampler)
        case _:
            logger.error(f"Unsupported sampler name: `{name}`")
            raise ValueError(
                f"Unsupported sampler name: `{name}`. "
                f"Choose one of the following: "
                f"{AVAILABLE_SAMPLERS}"
            )
    sampler = factory.get_sampler(name, **params)
    return sampler
