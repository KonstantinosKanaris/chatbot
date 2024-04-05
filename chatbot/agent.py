from typing import Any, Dict, List, Tuple

import mlflow
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from chatbot import logger
from chatbot.data.vocabulary import SequenceVocabulary
from chatbot.engine.trainer import Trainer
from chatbot.engine.utils import EarlyStopping, Timer, set_seeds
from chatbot.model.decoders import LuongAttnDecoderRNN
from chatbot.model.embeddings import EmbeddingLayerConstructor
from chatbot.model.encoders import EncoderRNN


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
        ...         "weight_decay": 0.001
        ...     },
        ...     "decoder_optimizer_init_params": {
        ...         "lr": 0.0001,
        ...         "weight_decay": 0.001
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
    a sequence-to-sequence model, from a saved checkpoint.

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
        ...         "temperature": 1.0
        ...     },
        ...     "encoder_optimizer_init_params": {
        ...         "lr": 0.0001,
        ...         "weight_decay": 0.001
        ...     },
        ...     "decoder_optimizer_init_params": {
        ...         "lr": 0.0001,
        ...         "weight_decay": 0.001,
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
    decoder.load_state_dict(state_dict=decoder_state_dict)

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


class Agent:
    """Controls the flow of the training process of a seq-to-seq
    model with PyTorch.

    Args:
        vocabulary (SequenceVocabulary): The dataset's vocabulary.
        hyperparameters (Dict[str, Any]): Set of hyperparameters.
        checkpoint_path (str): Path to save or load a checkpoint.
        resume (optional, bool): If ``True``, resumes training from
            a saved checkpoint. Defaults to ``False``.

    Attributes:
        vocabulary (SequenceVocabulary): The dataset's vocabulary.
        hyperparameters (Dict[str, Any]): Set of hyperparameters.
        checkpoint_path (str): Path to save checkpoint.
        resume (bool): If ``True``, resumes training from a saved
            checkpoint.
        training_params (Dict[str, Any]): Training parameters from
            the configuration file.
        early_stopper (EarlyStopping): Stops the training process
            if the validation loss does not decrease for a number
            of epochs.
    """

    def __init__(
        self,
        vocabulary: SequenceVocabulary,
        hyperparameters: Dict[str, Any],
        checkpoint_path: str,
        resume: bool = False,
    ) -> None:
        self.vocabulary = vocabulary
        self.hyperparameters = hyperparameters
        self.checkpoint_path = checkpoint_path
        self.resume = resume

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
        encoder_init_params = {
            f"encoder_{key}": val
            for key, val in {
                **self.hyperparameters["encoder_init_params"],
                **self.hyperparameters["encoder_optimizer_init_params"],
                **self.hyperparameters["encoder_lr_scheduler_init_params"],
            }.items()
        }

        decoder_init_params = {
            f"decoder_{key}": val
            for key, val in {
                **self.hyperparameters["decoder_init_params"],
                **self.hyperparameters["decoder_optimizer_init_params"],
                **self.hyperparameters["decoder_lr_scheduler_init_params"],
            }.items()
        }

        return {
            "vocab_size": len(self.vocabulary),
            **self.hyperparameters["general"],
            **self.hyperparameters["embedding_init_params"],
            **encoder_init_params,
            **decoder_init_params,
        }

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
                vocab_tokens=list(self.vocabulary.token_to_idx.keys()),
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
                vocab_tokens=list(self.vocabulary.token_to_idx.keys()),
                init_params=self.hyperparameters,
                use_optimizers=True,
            )

        loss_fn = nn.NLLLoss(ignore_index=self.vocabulary.mask_index)

        encoder_lr_scheduler = ReduceLROnPlateau(
            optimizer=training_components["encoder_optimizer"],
            mode="min",
            **self.hyperparameters["encoder_lr_scheduler_init_params"],
        )
        decoder_lr_scheduler = ReduceLROnPlateau(
            optimizer=training_components["decoder_optimizer"],
            mode="min",
            **self.hyperparameters["decoder_lr_scheduler_init_params"],
        )

        trainer = Trainer(
            embedding=training_components["embedding"],
            encoder=training_components["encoder"],
            decoder=training_components["decoder"],
            encoder_optimizer=training_components["encoder_optimizer"],
            decoder_optimizer=training_components["decoder_optimizer"],
            encoder_lr_scheduler=encoder_lr_scheduler,
            decoder_lr_scheduler=decoder_lr_scheduler,
            loss_fn=loss_fn,
            vocab=self.vocabulary,
            checkpoint_path=self.checkpoint_path,
            epochs=self.training_params["num_epochs"],
            last_epoch=last_epoch,
            clip_factor=self.training_params["clip_factor"],
            early_stopper=self.early_stopper,
            sampling_decay=self.training_params["sampling_decay"],
            sampling_method=self.training_params["sampling_method"],
            resume=self.resume,
            enable_early_stop=self.training_params["enable_early_stop"],
            enable_lr_scheduler=self.training_params["enable_lr_scheduler"],
        )
        return trainer, last_epoch

    def start_training(
        self,
        trainer: Trainer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        last_epoch: int = 0,
    ) -> None:
        """Splits the dataset into training and validation sets,
        creates dataloaders and begins the training process."""
        with mlflow.start_run():
            mlflow.log_params(params=self.training_params)

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
