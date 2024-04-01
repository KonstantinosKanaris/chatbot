from typing import Any, Dict, List, Tuple

import mlflow
import torch
from sklearn.model_selection import ShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm.auto import tqdm

from chatbot import logger
from chatbot.engine.trainer import Trainer
from chatbot.engine.utils import EarlyStopping, Timer, set_seeds
from chatbot.models.decoders import LuongAttnDecoderRNN
from chatbot.models.embeddings import EmbeddingLayerConstructor
from chatbot.models.encoders import EncoderRNN
from chatbot.utils.data.datasets import CornellDialogsDataset


def initialize_seq2seq_components(
    vocab_tokens: List[str],
    init_params: Dict[str, Any],
    use_optimizers: bool = False,
) -> Dict[str, Any]:
    """Creates the necessary components required for training or evaluating
    a sequence-to-sequence model.

    The components include:
        - Embedding layer
        - Encoder
        - Decoder
        - Encoder/Decoder optimizers (Used only for training)

    Args:
        vocab_tokens (List[str]): All vocabulary tokens. Used for creating
            the embedding layer and also specifies the output size of the
            decoder.
        init_params (Dict[str, Any]): Contains the initialization parameters
            for the embedding, encoder, and decoder layers and also for the
            optimizers.
        use_optimizers (bool, optional): If ``True`` returns the optimizers
            for training. Defaults to ``False``.

    Returns:
        Dict[str, Any]: A serializable dictionary containing the training
            components.

    Example:
        >>> # Load init_params either from a configuration (yaml) file
        >>> # or create it manually
        >>> seq2seq_params = {
        ...     "embeddings_path": "./pretrained_embeddings/glove.6B.50d.txt",
        ...     "embedding_init_params": {
        ...         "embedding_dim": 50,
        ...         "padding_idx": 0
        ...     }
        ...     "encoder_init_params": {
        ...         "hidden_size": 50,
        ...         "num_layers": 1,
        ...         "dropout": 0.1
        ...     }
        ...     "decoder_init_params": {
        ...         "alignment_method": "dot",
        ...         "hidden_size": 50,
        ...         "num_layers": 1,
        ...         "dropout": 0.1,
        ...     },
        ...     "encoder_optimizer_init_params": {
        ...         "lr": 0.0001,
        ...     },
        ...     "decoder_optimizer_init_params": {
        ...         "lr": 0.0001
        ...     }
        ... }
        >>>
        >>> training_components = initialize_seq2seq_components(
        ...     vocab_tokens=["1st_token", "2nd_token", "..."],
        ...     init_params=init_params,
        ...     use_optimizers=True
        ... )
        >>> embedding_layer = training_components["embedding"]
        >>> encoder_layer = training_components["encoder"]
        >>> decoder_layer = training_components["decoder"]
        >>> encoder_opt = training_components["encoder_optimizer"]
        >>> decoder_opt = training_components["decoder_optimizer"]
        >>> encoder_layer
        EncoderRNN(
          (emb): Embedding(2, 50, padding_idx=0)
          (bi_gru): GRU(50, 50, bidirectional=True)
        )
        >>> decoder_layer
        ...
    """
    embedding = EmbeddingLayerConstructor.create_embedding_layer(
        vocab_tokens=vocab_tokens,
        filepath=init_params["embeddings_path"],
        **init_params["embedding_init_params"],
    )

    encoder = EncoderRNN(
        embedding=embedding, **init_params["encoder_init_params"]
    )
    decoder = LuongAttnDecoderRNN(
        embedding=embedding,
        output_size=len(vocab_tokens),
        **init_params["decoder_init_params"],
    )

    encoder_optimizer, decoder_optimizer = None, None
    if use_optimizers:
        encoder_optimizer = torch.optim.Adam(
            params=encoder.parameters(),
            **init_params["encoder_optimizer_init_params"],
        )
        decoder_optimizer = torch.optim.Adam(
            params=decoder.parameters(),
            **init_params["decoder_optimizer_init_params"],
        )

    return {
        "embedding": embedding,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }


def initialize_from_checkpoint(
    path: str,
    vocab_tokens: List[str],
    init_params: Dict[str, Any],
    use_optimizers: bool = False,
) -> Dict[str, Any]:
    """Loads the necessary components required for training or evaluating
    a sequence-to-sequence model from a saved checkpoint.

    In addition to the model components, it also loads the last validation
    loss value and epoch, which are required to resume training.

    Note that the optimizers, loss value and epoch are required only for the
    training process and not for evaluation.

    Args:
        path (str): Path to the checkpoint.
        vocab_tokens (List[str]): All vocabulary tokens. Used for creating
            the embedding layer and also specifies the output size of the
            decoder.
        init_params (Dict[str, Any]): Contains the initialization parameters
            for the embedding, encoder, and decoder layers, as well as for the
            optimizers.
        use_optimizers (bool, optional): If ``True`` returns the optimizers
            for training. Defaults to ``False``.

    Returns:
        Dict[str, Any]: A serializable dictionary containing the training
            components.

    Example:
        >>> # Load init_params either from a configuration (yaml) file
        >>> # or create it manually
        >>> seq2seq_params = {
        ...     "embeddings_path": "./pretrained_embeddings/glove.6B.50d.txt",
        ...     "embedding_init_params": {
        ...         "embedding_dim": 50,
        ...         "padding_idx": 0
        ...     }
        ...     "encoder_init_params": {
        ...         "hidden_size": 50,
        ...         "num_layers": 1,
        ...         "dropout": 0.1
        ...     }
        ...     "decoder_init_params": {
        ...         "alignment_method": "dot",
        ...         "hidden_size": 50,
        ...         "num_layers": 1,
        ...         "dropout": 0.1,
        ...     },
        ...     "encoder_optimizer_init_params": {
        ...         "lr": 0.0001,
        ...     },
        ...     "decoder_optimizer_init_params": {
        ...         "lr": 0.0001
        ...     }
        ... }
        >>>
        >>> training_components = initialize_from_checkpoint(
        ...     path="./checkpoints/checkpoint_1.pth",
        ...     vocab_tokens=["1st_token", "2nd_token", "..."],
        ...     init_params=init_params,
        ...     use_optimizers=True
        ... )
        >>> embedding_layer = training_components["embedding"]
        >>> encoder_layer = training_components["encoder"]
        >>> decoder_layer = training_components["decoder"]
        >>> encoder_opt = training_components["encoder_optimizer"]
        >>> decoder_opt = training_components["decoder_optimizer"]
        >>> encoder_layer
        EncoderRNN(
          (emb): Embedding(2, 50, padding_idx=0)
          (bi_gru): GRU(50, 50, bidirectional=True)
        )
        >>> decoder_layer
        ...
    """
    try:
        checkpoint = torch.load(f=path)
    except RuntimeError:
        checkpoint = torch.load(
            f=path, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )

    embedding_state_dict = checkpoint["embedding"]
    encoder_state_dict = checkpoint["encoder"]
    decoder_state_dict = checkpoint["decoder"]
    encoder_optimizer_state_dict = checkpoint["encoder_optimizer"]
    decoder_optimizer_state_dict = checkpoint["decoder_optimizer"]

    embedding = EmbeddingLayerConstructor.create_embedding_layer(
        vocab_tokens=vocab_tokens,
        filepath=init_params["embeddings_path"],
        **init_params["embedding_init_params"],
    )
    embedding.load_state_dict(state_dict=embedding_state_dict)

    encoder = EncoderRNN(
        embedding=embedding, **init_params["encoder_init_params"]
    )
    decoder = LuongAttnDecoderRNN(
        embedding=embedding,
        output_size=len(vocab_tokens),
        **init_params["decoder_init_params"],
    )
    encoder.load_state_dict(state_dict=encoder_state_dict)
    decoder.load_state_dict(state_dict=decoder_state_dict, strict=False)

    encoder_optimizer, decoder_optimizer = None, None
    if use_optimizers:
        encoder_optimizer = torch.optim.Adam(
            params=encoder.parameters(),
            **init_params["encoder_optimizer_init_params"],
        )
        decoder_optimizer = torch.optim.Adam(
            params=decoder.parameters(),
            **init_params["decoder_optimizer_init_params"],
        )
        encoder_optimizer.load_state_dict(
            state_dict=encoder_optimizer_state_dict
        )
        decoder_optimizer.load_state_dict(
            state_dict=decoder_optimizer_state_dict
        )

    return {
        "embedding": embedding,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
        "last_best_score": checkpoint["val_loss"],
        "last_epoch": checkpoint["epoch"],
    }


class TrainingController:
    """Controls the flow of the training process of a seq-to-seq
    model with PyTorch.

    The flow includes:

    .. code-block:: text

        - Initialization of encoder and decoder layers
        - Setting up encoder and decoder optimizers
        - Splitting data into validation and training sets
        - Creation of dataloaders
        - Start of training process

    Args:
        dataset (CornellDialogsDataset): The cornell dialogs dataset.
        hyperparameters (Dict[str, Any]): Set of hyperparameters.
        checkpoint_path (str): Path to save or load a checkpoint.
        resume (optional, bool): If ``True``, resumes training from
            a saved checkpoint. Defaults to ``False``.

    Attributes:
        dataset (CornellDialogsDataset): The cornell dialogs dataset.
        hyperparameters (Dict[str, Any]): Set of hyperparameters.
        checkpoint_path (str): Path to save checkpoint.
        resume (bool): If ``True``, resumes training from a saved
            checkpoint.
        vectorizer (SequenceVectorizer): Class responsible for converting
            tokens to numbers.

    Example:
        >>> # Create a `Dataset` object from the dataset's txt file
        >>> # and with a `split()` tokenizer
        >>> from chatbot.utils.data.datasets import CornellDialogsDataset
        >>> from chatbot.utils.aux import get_tokenizer
        >>>
        >>> split_tokenizer = get_tokenizer(tokenizer=None)
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
        ...     "embeddings_path": "./pretrained_embeddings/glove.6B.50d.txt",
        ...     "data": {
        ...         "min_seq_length": 1,
        ...         "max_seq_length": 10,
        ...         "min_count": 3,
        ...         "validation_size": 0.3,
        ...         "batch_size": 64
        ...     },
        ...     "general": {
        ...         "num_epochs": 10,
        ...         "batch_size": 64,
        ...         "lr_patience": 3,
        ...         "lr_reduce_factor": 0.25,
        ...         "ea_patience": 7,
        ...         "ea_delta": 0.005,
        ...         "clip_factor": 50,
        ...         "enable_early_stop": True,
        ...         "sampling_decay": 10
        ...     },
        ...     "embedding_init_params": {
        ...         "embedding_dim": 50,
        ...         "padding_idx": 0
        ...     }
        ...     "encoder_init_params": {
        ...         "hidden_size": 50,
        ...         "num_layers": 1,
        ...         "dropout": 0.1
        ...     }
        ...     "decoder_init_params": {
        ...         "alignment_method": "dot",
        ...         "hidden_size": 50,
        ...         "num_layers": 1,
        ...         "dropout": 0.1,
        ...     },
        ...     "encoder_optimizer_init_params": {
        ...         "lr": 0.0001,
        ...     },
        ...     "decoder_optimizer_init_params": {
        ...         "lr": 0.0001
        ...     }
        ... }
        >>>
        >>> # Initialize the training controller
        >>> training_controller = TrainingController(
        ...     dataset=dataset,
        ...     hyperparameters=hyperparameters,
        ...     checkpoints_path="./checkpoints/checkpoint_1.pth"
        ... )
        >>>
        >>> # Start training process
        >>> trainer, last_epoch = training_controller.prepare_for_training()
        >>> training_controller.start_training(
        ...     trainer=trainer, last_epoch=last_epoch
        ... )
    """

    def __init__(
        self,
        dataset: CornellDialogsDataset,
        hyperparameters: Dict[str, Any],
        checkpoint_path: str,
        resume: bool = False,
    ) -> None:
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.checkpoint_path = checkpoint_path
        self.resume = resume

        self.vectorizer = dataset.get_vectorizer()
        self.training_params = self._load_training_params()
        self.early_stopper = EarlyStopping(
            patience=self.training_params["ea_patience"],
            delta=self.training_params["ea_delta"],
            verbose=True,
            path=checkpoint_path,
        )

        set_seeds(seed=42)

    def _load_training_params(self) -> Dict[str, Any]:
        """Returns all the training hyperparameters from the
        configuration file."""
        return {
            "vocab_size": len(self.vectorizer.vocab),
            **self.hyperparameters["data"],
            **self.hyperparameters["general"],
            **self.hyperparameters["embedding_init_params"],
            **self.hyperparameters["encoder_init_params"],
            **self.hyperparameters["decoder_init_params"],
            **self.hyperparameters["encoder_optimizer_init_params"],
            **self.hyperparameters["decoder_optimizer_init_params"],
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

    def prepare_for_training(self) -> Tuple[Trainer, int]:
        """Loads the training components, including the embedding,
        encoder, and decoder layers along with the encoder and
        decoder optimizers.

        If :attr:`resume` is ``True`` the training components are
        initialized with their saved states from the last checkpoint.

        Returns:
             A `Trainer` object responsible for executing the training
            process and the last epoch.
        """
        if self.resume:
            training_components = initialize_from_checkpoint(
                path=self.checkpoint_path,
                vocab_tokens=list(self.vectorizer.vocab.token_to_idx.keys()),
                init_params=self.hyperparameters,
                use_optimizers=True,
            )
            last_epoch = training_components["last_epoch"]
            last_best_score = training_components["last_best_score"]
            logger.info(
                f"Resume training from checkpoint: {self.checkpoint_path}"
            )
            logger.info(f"Last validation loss value: {last_best_score:.4f}\n")
            logger.info(f"Resuming from epoch {last_epoch + 1}\n")
            self.early_stopper.val_loss_min = last_best_score
            self.early_stopper.best_score = -last_best_score
        else:
            last_epoch = 0
            training_components = initialize_seq2seq_components(
                vocab_tokens=list(self.vectorizer.vocab.token_to_idx.keys()),
                init_params=self.hyperparameters,
                use_optimizers=True,
            )

        loss_fn = nn.NLLLoss(ignore_index=self.vectorizer.vocab.mask_index)

        trainer = Trainer(
            embedding=training_components["embedding"],
            encoder=training_components["encoder"],
            decoder=training_components["decoder"],
            encoder_optimizer=training_components["encoder_optimizer"],
            decoder_optimizer=training_components["decoder_optimizer"],
            loss_fn=loss_fn,
            vocab=self.vectorizer.vocab,
            checkpoint_path=self.checkpoint_path,
            epochs=self.training_params["num_epochs"],
            last_epoch=last_epoch,
            clip_factor=self.training_params["clip_factor"],
            early_stopper=self.early_stopper,
            sampling_decay=self.training_params["sampling_decay"],
            resume=self.resume,
            enable_early_stop=self.training_params["enable_early_stop"],
        )
        return trainer, last_epoch

    def start_training(self, trainer: Trainer, last_epoch: int = 0) -> None:
        """Splits the dataset into training and validation sets,
        creates dataloaders and begins the training process."""
        with mlflow.start_run():
            mlflow.log_params(params=self.training_params)

            rs = ShuffleSplit(
                n_splits=1,
                test_size=self.training_params["validation_size"],
                random_state=42,
            )

            logger.info(
                "------------------------- Training -------------------------"
            )
            logger.info(
                f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
            )

            bar_fmt = (
                "{desc} {percentage:3.0f}%|{bar}{postfix} "
                "[{elapsed}<{remaining}]"
            )
            try:
                for fold, (train_ids, val_ids) in enumerate(
                    rs.split(X=self.dataset)
                ):
                    logger.info(
                        f"Training size: {len(train_ids)}, "
                        f"Validation size: {len(val_ids)}\n"
                    )

                    train_dataloader, val_dataloader = (
                        self._create_dataloaders(
                            train_ids=train_ids,
                            val_ids=val_ids,
                            batch_size=self.training_params["batch_size"],
                        )
                    )
                    tqdm_bar = tqdm(
                        iterable=range(self.training_params["num_epochs"] + 1),
                        total=self.training_params["num_epochs"],
                        bar_format=bar_fmt,
                        initial=last_epoch + 1,
                        position=0,
                        leave=False,
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
