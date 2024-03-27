import argparse
import os

import torch
from torch import nn

from chatbot import __title__, logger
from chatbot.controllers import TrainingController
from chatbot.datasets import CornellDialogsDataset
from chatbot.engine.evaluator import Evaluator, GreedySearchDecoder
from chatbot.models.decoders import LuongAttnDecoderRNN
from chatbot.models.embeddings import PreTrainedEmbeddings
from chatbot.models.encoders import EncoderRNN
from chatbot.utils.aux import get_tokenizer, load_yaml_file


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
        help="If `yes` the training will resume from the last saved checkpoint.",
    )

    evaluate.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    data_path = "./data/cornell_movie_dialogs/processed/formatted_dialogs.txt"

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
        training_controller = TrainingController(
            dataset=dataset,
            hyperparameters=config["hyperparameters"],
            checkpoints_dir=config["checkpoints_dir"],
            resume=resume,
        )
        trainer = training_controller.prepare_for_training()
        training_controller.start_training(trainer=trainer)
    elif arguments.mode == "evaluate":
        dataset = CornellDialogsDataset.load_pairs_and_vectorizer(
            file=config["data_path"],
            tokenizer=tokenizer,
            max_length=config["max_seq_length"] + 1,
            min_count=config["min_count"] + 1,
        )
        vectorizer = dataset.get_vectorizer()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(
            f=config["checkpoint_path"], map_location=torch.device("cpu")
        )
        embedding_state_dict = checkpoint["embedding"]
        encoder_state_dict = checkpoint["encoder"]
        decoder_state_dict = checkpoint["decoder"]

        embeddings_path = config["embeddings_path"]
        if embeddings_path and os.path.exists(embeddings_path):
            embeddings_wrapper = PreTrainedEmbeddings.from_embeddings_file(
                embedding_file=embeddings_path
            )
            input_embeddings = embeddings_wrapper.make_embedding_matrix(
                words=list(vectorizer.vocab.token_to_idx.keys())
            )
            embedding = nn.Embedding(
                num_embeddings=vectorizer.vocab.num_tokens,
                padding_idx=0,
                _weight=input_embeddings,
                **config["embedding_init_params"],
            )
        else:
            embedding = nn.Embedding(
                num_embeddings=vectorizer.vocab.num_tokens,
                padding_idx=0,
                **config["embedding_init_params"],
            )

        embedding.load_state_dict(embedding_state_dict)

        encoder = EncoderRNN(embedding=embedding, **config["encoder_init_params"])
        decoder = LuongAttnDecoderRNN(
            embedding=embedding,
            output_size=vectorizer.vocab.num_tokens,
            **config["decoder_init_params"],
        )
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)

        encoder = encoder.to(device)
        decoder = decoder.to(device)

        searcher = GreedySearchDecoder(encoder=encoder, decoder=decoder)
        evaluator = Evaluator(vectorizer=vectorizer)
        # evaluator.evaluate("hello how are you today?", searcher=searcher)
        evaluator.chat(searcher=searcher)
    else:
        logger.error("Not supported mode.")
