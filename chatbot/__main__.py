import argparse
import os

import torch

from chatbot import __title__, logger
from chatbot.controllers import TrainingController, initialize_from_checkpoint
from chatbot.engine.evaluator import Evaluator
from chatbot.models.decoders import GreedySearchDecoder
from chatbot.utils.aux import get_tokenizer, load_yaml_file
from chatbot.utils.data.datasets import CornellDialogsDataset


def parse_arguments() -> argparse.Namespace:
    """
    Constructs parsers and subparsers.

    Returns:
        argparse.Namespace:
            The parser/subparser with its arguments.
    """
    parser = argparse.ArgumentParser(
        description=f"Command Line Interface for {__title__}"
    )

    subparsers = parser.add_subparsers(
        description="Project functionalities", dest="mode"
    )

    train = subparsers.add_parser(
        name="train", help="This is the subparser for training."
    )

    evaluate = subparsers.add_parser(
        name="evaluate", help="This is the subparser for evaluation."
    )

    train.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for training."
        "Contains data paths, hyperparameters, etc.",
    )

    train.add_argument(
        "--resume_from_checkpoint",
        type=str,
        choices={"yes", "no"},
        required=False,
        default="no",
        help="If `yes` the training will " "resume from the last saved checkpoint.",
    )

    evaluate.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    config = load_yaml_file(filepath=arguments.config)
    tokenizer = get_tokenizer()

    if arguments.mode == "train":
        if arguments.resume_from_checkpoint == "yes":
            resume = True
        else:
            resume = False
        dataset = CornellDialogsDataset.load_pairs_and_vectorizer(
            file=config["data_path"],
            tokenizer=tokenizer,
            max_length=config["hyperparameters"]["general"]["max_seq_length"] + 1,
            min_count=config["hyperparameters"]["general"]["min_count"] + 1,
        )

        checkpoint_path = str(
            os.path.join(config["checkpoints_dir"], config["checkpoint_filename"])
        )
        training_controller = TrainingController(
            dataset=dataset,
            hyperparameters=config["hyperparameters"],
            checkpoint_path=checkpoint_path,
            resume=resume,
        )
        trainer, last_epoch = training_controller.prepare_for_training()
        training_controller.start_training(trainer=trainer, last_epoch=last_epoch)
    elif arguments.mode == "evaluate":
        dataset = CornellDialogsDataset.load_pairs_and_vectorizer(
            file=config["data_path"],
            tokenizer=tokenizer,
            max_length=config["evaluation_parameters"]["general"]["max_seq_length"] + 1,
            min_count=config["evaluation_parameters"]["general"]["min_count"] + 1,
        )
        vectorizer = dataset.get_vectorizer()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        evaluation_components = initialize_from_checkpoint(
            path=config["checkpoint_path"],
            vocab_tokens=list(vectorizer.vocab.token_to_idx.keys()),
            init_params=config["evaluation_parameters"],
            use_optimizers=False,
        )

        encoder = evaluation_components["encoder"].to(device)
        decoder = evaluation_components["decoder"].to(device)

        searcher = GreedySearchDecoder(
            encoder=encoder,
            decoder=decoder,
            eos_idx=vectorizer.vocab.end_seq_index,
        )
        evaluator = Evaluator(vectorizer=vectorizer)
        # decoded_words = evaluator.evaluate("hello?", searcher=searcher)
        evaluator.chat(searcher=searcher)
    else:
        logger.error("Not supported mode.")
