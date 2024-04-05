import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

from chatbot import __title__, logger
from chatbot.agent import Agent, initialize_from_checkpoint
from chatbot.data.datasets import CornellDialogsDataset
from chatbot.data.download import get_data
from chatbot.data.parsers import CornellDialogsParser
from chatbot.data.processing import Seq2SeqProcessor
from chatbot.data.tokenizers import get_tokenizer
from chatbot.data.utils import (
    load_pairs,
    load_txt_file,
    load_yaml_file,
    save_pairs,
)
from chatbot.data.vectorizer import SequenceVectorizer
from chatbot.data.vocabulary import build_vocabulary
from chatbot.engine.evaluator import Evaluator
from chatbot.model.clients import sampler_client


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

    download = subparsers.add_parser(
        name="download", help="This is the subparser for downloading data."
    )
    parse = subparsers.add_parser(
        name="parse", help="This is the subparser for parsing raw data."
    )
    process = subparsers.add_parser(
        name="process",
        help="This is the subparser for processing the"
        "sentences pairs in the parsed `.txt. file"
        "created by the `parse` subparser.",
    )
    create_assets = subparsers.add_parser(
        name="create_assets",
        help="This is the subparser for creating training, validation"
        "and test splits from the processed `txt` file",
    )
    train = subparsers.add_parser(
        name="train", help="This is the subparser for training."
    )
    evaluate = subparsers.add_parser(
        name="evaluate", help="This is the subparser for evaluation."
    )

    download.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the dataset to download."
        "Available datasets: `cornell_dialogs`.",
    )
    download.add_argument(
        "--directory",
        type=str,
        required=True,
        help="A local directory path to save the data.",
    )

    parse.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Filepath to the raw data for parsing.",
    )
    parse.add_argument(
        "--delimiter",
        type=str,
        required=True,
        help="Delimiter that separates the query sequence from the response.",
    )
    parse.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Filepath to save the parsed data.",
    )

    process.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Filepath to the parsed `txt` file.",
    )
    process.add_argument(
        "--min_length",
        type=int,
        required=False,
        default=1,
        help="Minimum number of tokens allowed in each of the"
        "pair of sequences (default=1).",
    )
    process.add_argument(
        "--max_length",
        type=int,
        required=False,
        default=10,
        help="Maximum number of tokens allowed in each of the"
        "pair of sequences (default=1)",
    )
    process.add_argument(
        "--min_count",
        type=int,
        required=False,
        default=3,
        help="Minimum token count threshold (default=3).",
    )
    process.add_argument(
        "--delimiter",
        type=str,
        required=True,
        help="Delimiter that separates the query sequence from the response.",
    )
    process.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Filepath to save the processed data.",
    )

    create_assets.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Filepath to the processed `txt` file.",
    )
    create_assets.add_argument(
        "--val_size",
        type=float,
        required=True,
        help="Float number between 0.0 and 1.0 representing the proportion "
        "of the dataset to include in the validation split.",
    )
    create_assets.add_argument(
        "--test_size",
        type=float,
        required=True,
        help="Float number between 0.0 and 1.0 representing the proportion "
        "of the dataset to include in the test split.",
    )
    create_assets.add_argument(
        "--random_seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for the splitting process to "
        "ensure reproducibility.",
    )
    create_assets.add_argument(
        "--assets_directory",
        type=str,
        required=True,
        help="Directory to save the training, validation and test data.",
    )

    train.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for training."
        "Contains data paths, hyperparameters, etc.",
    )
    train.add_argument(
        "--delimiter",
        type=str,
        required=True,
        help="Delimiter that separates the query sequence from the response.",
    )
    train.add_argument(
        "--resume_from_checkpoint",
        type=str,
        choices={"yes", "no"},
        required=False,
        default="no",
        help="If `yes` the training will "
        "resume from the last saved checkpoint.",
    )

    evaluate.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for evaluation.",
    )
    evaluate.add_argument(
        "--delimiter",
        type=str,
        required=True,
        help="Delimiter that separates the query sequence from the response.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    tokenizer = get_tokenizer()

    if arguments.mode == "download":
        get_data(name=arguments.name, directory=arguments.directory)
    elif arguments.mode == "parse":
        conversations = CornellDialogsParser.parse_conversations(
            filepath=arguments.filepath
        )
        list_of_pairs = CornellDialogsParser.extract_sequence_pairs(
            conversations=conversations
        )
        save_pairs(
            pairs=list_of_pairs,
            path=arguments.save_path,
            delimiter=arguments.delimiter,
        )
    elif arguments.mode == "process":
        processor = Seq2SeqProcessor.from_txt_file(
            file=arguments.filepath,
            tokenizer=tokenizer,
            delimiter=arguments.delimiter,
        )
        filtered_pairs = processor.filter_pairs_by_length(
            pairs=processor.pairs,
            min_length=arguments.min_length,
            max_length=arguments.max_length,
        )
        logger.info(
            f"Filtered (based on sequence length) to "
            f"{len(filtered_pairs)} sequence pairs."
        )
        filtered_pairs = processor.filter_pairs_by_token_count(
            pairs=filtered_pairs, min_count=arguments.min_count
        )
        logger.info(
            f"Filtered (based on token count) to "
            f"{len(filtered_pairs)} sequence pairs."
        )
        save_pairs(
            pairs=filtered_pairs,
            path=arguments.save_path,
            delimiter=arguments.delimiter,
        )
    elif arguments.mode == "create_assets":
        pairs = load_txt_file(filepath=arguments.filepath)
        logger.info("Splitting data...")
        random.Random(arguments.random_seed).shuffle(pairs)

        val_num_indices = int(arguments.val_size * len(pairs))
        test_num_indices = int(arguments.test_size * len(pairs))

        val_set = pairs[:val_num_indices]
        test_set = pairs[
            val_num_indices : val_num_indices + test_num_indices  # noqa: E203
        ]
        train_set = pairs[val_num_indices + test_num_indices :]  # noqa: E203

        logger.info(
            f"train_set size: {len(train_set)}, "
            f"val_set size: {len(val_set)}, "
            f"test_set size: {len(test_set)}"
        )

        assets = [train_set, val_set, train_set]
        assets_filepaths = [
            os.path.join(arguments.assets_directory, "train_set.txt"),
            os.path.join(arguments.assets_directory, "val_set.txt"),
            os.path.join(arguments.assets_directory, "test_set.txt"),
        ]

        if not os.path.isdir(arguments.assets_directory):
            os.makedirs(arguments.assets_directory, exist_ok=True)

        for data_set, data_path in list(zip(assets, assets_filepaths)):
            filename = os.path.splitext(os.path.basename(data_path))[0]
            logger.info(f"Saving {filename} to {data_path}")
            with open(file=data_path, mode="w", encoding="utf-8") as f:
                for line in data_set:
                    f.write(f"{line}\n")
    elif arguments.mode == "train":
        config = load_yaml_file(filepath=arguments.config)

        if arguments.resume_from_checkpoint == "yes":
            resume = True
        else:
            resume = False

        train_pairs = load_pairs(
            filepath=config["train_data_path"], delimiter=arguments.delimiter
        )
        val_pairs = load_pairs(
            filepath=config["val_data_path"], delimiter=arguments.delimiter
        )

        vocab, max_query_len, max_response_len = build_vocabulary(
            pairs=train_pairs, tokenizer=tokenizer
        )

        vectorizer = SequenceVectorizer.from_serializable(
            contents={
                "vocabulary": vocab,
                "tokenizer": tokenizer,
                "max_query_length": max_query_len,
                "max_response_length": max_response_len,
            }
        )

        train_dataset = CornellDialogsDataset(
            pairs=train_pairs, vectorizer=vectorizer
        )
        val_dataset = CornellDialogsDataset(
            pairs=val_pairs, vectorizer=vectorizer
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config["hyperparameters"]["general"]["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config["hyperparameters"]["general"]["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        checkpoint_path = str(
            os.path.join(
                config["checkpoints_dir"], config["checkpoint_filename"]
            )
        )

        training_controller = Agent(
            vocabulary=vocab,
            hyperparameters=config["hyperparameters"],
            checkpoint_path=checkpoint_path,
            resume=resume,
        )
        trainer, last_epoch = training_controller.prepare_for_training()
        training_controller.start_training(
            trainer=trainer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            last_epoch=last_epoch,
        )
    elif arguments.mode == "evaluate":
        config = load_yaml_file(filepath=arguments.config)
        train_pairs = load_pairs(
            filepath=config["train_data_path"], delimiter=arguments.delimiter
        )
        vocab, max_query_len, max_response_len = build_vocabulary(
            pairs=train_pairs, tokenizer=tokenizer
        )

        vectorizer = SequenceVectorizer.from_serializable(
            contents={
                "vocabulary": vocab,
                "tokenizer": tokenizer,
                "max_query_length": max_query_len,
                "max_response_length": max_response_len,
            }
        )

        train_dataset = CornellDialogsDataset(
            pairs=train_pairs, vectorizer=vectorizer
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        evaluation_components = initialize_from_checkpoint(
            path=config["checkpoint_path"],
            vocab_tokens=list(vectorizer.vocab.token_to_idx.keys()),
            init_params=config["evaluation_parameters"],
            use_optimizers=False,
        )

        encoder = evaluation_components["encoder"].to(device)
        decoder = evaluation_components["decoder"].to(device)

        sampler = sampler_client(
            name=config["evaluation_parameters"]["sampler"],
            params={
                "encoder": encoder,
                "decoder": decoder,
                "eos_idx": vectorizer.vocab.end_seq_index,
            },
        )
        evaluator = Evaluator(vectorizer=vectorizer)
        evaluator.chat(searcher=sampler)
    else:
        logger.error("Not supported mode.")
