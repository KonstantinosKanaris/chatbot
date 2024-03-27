import os
from datetime import datetime
from typing import Any, Dict, Tuple

import mlflow
import torch
from sklearn.model_selection import ShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm

from chatbot import logger
from chatbot.datasets import CornellDialogsDataset
from chatbot.engine.trainer import Trainer
from chatbot.engine.utils import Timer, set_seeds
from chatbot.models.decoders import LuongAttnDecoderRNN
from chatbot.models.embeddings import PreTrainedEmbeddings
from chatbot.models.encoders import EncoderRNN


class TrainingController:
    """Controls the flow of the training process of a seq-to-seq
    conversational model with PyTorch.

    The entire process in high-level includes:

    .. code-block:: text

        - Initialization of encoder and decoder layers
        - Setting up encoder and decoder optimizers
        - Splitting data into validation and training sets
        - Creation of dataloaders
        - Start training process

    Args:
        dataset (CornellDialogsDataset): The cornell dialogs dataset.
        hyperparameters (Dict[str, Any]): Set of hyperparameters.
        checkpoints_dir (str): Directory to save checkpoints.
        resume (optional, bool): If ``True``, resumes training from
            a saved checkpoint. Defaults to ``False``.

    Attributes:
        dataset (CornellDialogsDataset): The cornell dialogs dataset.
        hyperparameters (Dict[str, Any]): Set of hyperparameters.
        checkpoints_dir (str): Directory to save checkpoints.
        resume (bool): If ``True``, resumes training from a saved
            checkpoint.
        vectorizer (SequenceVectorizer): Class responsible for converting
            tokens to numbers.
        mask_index (int): Index of the mask token in the vocabulary.

    Example:
        >>> # Create a `Dataset` object from the dataset's txt file
        >>> # and with a `split()` tokenizer
        >>> from chatbot.datasets import CornellDialogsDataset
        >>> from chatbot.utils.aux import get_tokenizer
        >>> split_tokenizer = get_tokenizer(tokenizer=None) # If None returns split()
        >>> dataset = CornellDialogsDataset.load_pairs_and_vectorizer(
        ...     file="./cornel_dialogs.txt",
        ...     tokenizer=split_tokenizer,
        ...     min_count=3,
        ...     max_length=10
        ... )
        >>>
        >>> # Define training hyperparameters in the following format:
        >>> # Note that the hyperparameters are defined in a yaml file
        >>> # but with the same format.
        >>> hyperparameters = {
        ...     "general": {
        ...         "num_epochs": 10,
        ...         "batch_size": 64,
        ...         "lr_patience": 3,
        ...         "lr_reduce_factor": 0.25,
        ...         "ea_patience": 7,
        ...         "ea_delta": 0.005,
        ...         "clip_factor": 50,
        ...         "max_seq_length": 10,
        ...         "min_count": 3
        ...     },
        ...     "model_init_params": {
        ...         "alignment_method": "dot",
        ...         "embedding_dim": 50,
        ...         "hidden_size": 50,
        ...         "dropout": 0.1,
        ...         "encoder_num_layers": 1,
        ...         "decoder_num_layers": 1,
        ...         "teacher_forcing_ration": 0
        ...     },
        ...     "optimizer_init_params": {
        ...         "encoder_lr": 0.0001,
        ...         "decoder_lr": 0.0001
        ...     }
        ... }
        >>>
        >>> # Initialize the training controller
        >>> training_controller = TrainingController(
        ...     dataset=dataset,
        ...     hyperparameters=hyperparameters,
        ...     checkpoints_dir="./checkpoints/"
        ... )
        >>>
        >>> # Start training process
        >>> trainer = training_controller.prepare_for_training()
        >>> training_controller.start_training(trainer=trainer)
    """

    def __init__(
        self,
        dataset: CornellDialogsDataset,
        hyperparameters: Dict[str, Any],
        checkpoints_dir: str,
        resume: bool = False,
    ) -> None:
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.checkpoints_dir = checkpoints_dir
        self.resume = resume

        self.vectorizer = dataset.get_vectorizer()
        self.mask_index = self.vectorizer.vocab.mask_index
        self.training_params = self._load_training_params()

    def _load_training_params(self) -> Dict[str, Any]:
        """Returns all the training hyperparameters from the
        configuration file."""
        return {
            **self.hyperparameters["general"],
            **self.hyperparameters["model_init_params"],
            **self.hyperparameters["optimizer_init_params"],
        }

    def _create_dataloaders(
        self, train_ids, val_ids, batch_size
    ) -> Tuple[DataLoader, DataLoader]:
        """Creates training and validation dataloaders from
        a sample of training and validation indices.

        Args:
            train_ids (numpy.ndarray): Sample of training indices.
            val_ids (numpy.ndarray): Sample of validation indices.
            batch_size (int): Number of samples per batch to load.

        Returns:
            Tuple[DataLoader, DataLoader]: The training
                and validation dataloaders.
        """
        train_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_ids),
        )
        val_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(val_ids),
        )
        return train_dataloader, val_dataloader

    def prepare_for_training(self) -> Trainer:
        """Defines the layers of the seq-to-seq model, i.e., the
        embedding, encoder, and decoder layers.
        #TODO
        """
        set_seeds(seed=42)

        embeddings_path = self.hyperparameters["embeddings_path"]
        if embeddings_path and os.path.exists(embeddings_path):
            embeddings_wrapper = PreTrainedEmbeddings.from_embeddings_file(
                embedding_file=embeddings_path
            )
            input_embeddings = embeddings_wrapper.make_embedding_matrix(
                words=list(self.vectorizer.vocab.token_to_idx.keys())
            )
            embedding = nn.Embedding(
                num_embeddings=self.vectorizer.vocab.num_tokens,
                embedding_dim=self.training_params["embedding_dim"],
                padding_idx=0,
                _weight=input_embeddings,
            )
        else:
            embedding = nn.Embedding(
                num_embeddings=self.vectorizer.vocab.num_tokens,
                embedding_dim=self.training_params["embedding_dim"],
                padding_idx=0,
            )

        encoder = EncoderRNN(
            embedding=embedding,
            hidden_size=self.training_params["hidden_size"],
            num_layers=self.training_params["encoder_num_layers"],
            dropout=self.training_params["dropout"],
        )
        decoder = LuongAttnDecoderRNN(
            alignment_method="dot",
            embedding=embedding,
            hidden_size=self.training_params["hidden_size"],
            output_size=self.vectorizer.vocab.num_tokens,
            num_layers=self.training_params["decoder_num_layers"],
            dropout=self.training_params["dropout"],
        )

        encoder_optimizer = torch.optim.Adam(
            params=encoder.parameters(), lr=self.training_params["encoder_lr"]
        )
        decoder_optimizer = torch.optim.Adam(
            params=decoder.parameters(), lr=self.training_params["decoder_lr"]
        )

        if self.hyperparameters["checkpoint_filename"]:
            checkpoint_path = os.path.join(
                self.checkpoints_dir, self.hyperparameters["checkpoint_filename"]
            )
        else:
            current_datetime = datetime.now().strftime("%Y%m%d_%H-%M")
            checkpoint_path = os.path.join(
                self.checkpoints_dir,
                f"{current_datetime}_checkpoint.pth",
            )

        trainer = Trainer(
            embedding=embedding,
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            vectorizer=self.vectorizer,
            checkpoint_path=checkpoint_path,
            epochs=self.training_params["num_epochs"],
            patience=self.training_params["ea_patience"],
            delta=self.training_params["ea_delta"],
            clip_factor=self.training_params["clip_factor"],
            teacher_forcing_ratio=self.training_params["teacher_forcing_ratio"],
            resume=self.resume,
        )
        return trainer

    def start_training(self, trainer: Trainer) -> None:
        """Splits the dataset into training and validation sets,
        creates dataloaders and begins the training process."""
        with mlflow.start_run():
            mlflow.log_params(params=self.training_params)

            rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            logger.info(
                "-------------------------- Training --------------------------"
            )
            logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

            bar_format = (
                "{desc} {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]"
            )
            try:
                for fold, (train_ids, val_ids) in enumerate(rs.split(X=self.dataset)):
                    train_dataloader, val_dataloader = self._create_dataloaders(
                        train_ids=train_ids,
                        val_ids=val_ids,
                        batch_size=self.training_params["batch_size"],
                    )
                    tqdm_bar = tqdm(
                        bar_format=bar_format, initial=1, position=0, leave=False
                    )
                    with Timer() as t:
                        trainer.train(
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            tqdm_bar=tqdm_bar,
                        )
                    logger.info(f"Training took {t.elapsed} seconds.\n")
            except KeyboardInterrupt:
                logger.info("Exiting loop.\n")
