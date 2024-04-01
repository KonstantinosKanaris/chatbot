import random
from collections.abc import Iterator
from typing import Any, Dict, Optional

import mlflow
import torch
import torch.utils.data
from prefetch_generator import BackgroundGenerator
from torch import nn
from tqdm import tqdm

from chatbot import logger
from chatbot.engine.utils import inverse_sigmoid_decay, write_checkpoint
from chatbot.utils.data.vocabulary import SequenceVocabulary


class Trainer:
    """Class for training a sequence-to-sequence text generation model with
    PyTorch.

    The model consists of encoder and decoder layers, each trained separately
    using its own optimizer.

    Incorporates functionalities such as early stopping, resuming from
    checkpoint, learning rate scheduling and MLFlow tracking.

    Args:
        checkpoint_path (str): The file path to save or load a checkpoint.
        embedding (torch.nn.Embedding): The embedding layer.
        encoder (torch.nn.Module): The encoder layer.
        decoder (torch.nn.Module): The decoder layer.
        encoder_optimizer (torch.optim.Optimizer): The optimizer for updating
            the parameters of the encoder.
        decoder_optimizer (torch.optim.Optimizer): The optimizer for updating
            the parameters of the decoder.
        loss_fn (torch.nn.Model): Loss to optimize.
        vocab (SequenceVocabulary): The dataset's vocabulary.
        epochs (int, optional): Number of training epochs (default=5).
        clip_factor (float, optional): Max norm value for gradient clipping.
            Defaults to ``None``.
        resume (bool, optional): If ``True``, resumes training from
            the specified checkpoint. Defaults to ``False``.
        enable_early_stop (bool, optional): If ``True`` utilizes the
            :attr:`early_stopper` for stopping the training process
            if the validation loss doesn't decrease for a number of epochs.
            Defaults to ``False``.
        last_epoch (int, optional): Used in case of resuming training
            from the last checkpoint (default=0).
        sampling_decay (int | float, optional): Decay value for the
            sampling scheduler. Refer to
            :func:`chatbot.engine.utils.inverse_sigmoid_decay`.
        early_stopper (EarlyStopping, optional): Stops the training process
            if the validation loss doesn't decrease for a number of epochs.
            Defaults to ``None``.
        scheduler (ReduceLROnPlateau, optional): Reduces the learning rate when
            the validation loss stops improving. Defaults to ``None``.
    """

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        vocab: SequenceVocabulary,
        checkpoint_path: str,
        epochs: int = 5,
        clip_factor: Optional[float | int] = None,
        resume: bool = False,
        enable_early_stop: bool = False,
        last_epoch: int = 0,
        sampling_decay: int | float = 1.5,
        early_stopper: Optional[Any] = None,
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    ) -> None:

        if enable_early_stop and not early_stopper:
            raise TypeError("enable_early_stop is True but early_stopper is None!")

        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.loss_fn = loss_fn
        self.vocab = vocab
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.resume = resume
        self.enable_early_stop = enable_early_stop
        self.last_epoch = last_epoch
        self.sampling_decay = sampling_decay
        self.clip_factor = clip_factor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.early_stopper = early_stopper
        self.sampling_probability: float = 1.0

    def compute_metrics(
        self,
        decoder_input: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_state: torch.Tensor,
        target_sequences: torch.Tensor,
        target_masks: torch.Tensor,
        target_max_length: int,
    ) -> torch.Tensor:
        """Performs a forward pass through the decoder.

        For each token in the input sequence the loss is computed
        and accumulated.

        Implements a scheduled sampling mechanism which, based on
        the :attr:`sampling_probability`, decides whether to use
        the actual target token as next input to the decoder or
        an estimated token coming from the decoder's output.

        The estimation from the decoder is obtained by sampling a token
        according to decoder's output probability distribution over the
        vocabulary.

        Args:
            decoder_input: tensor of shape: math:`(1, N)`.
            decoder_hidden: tensor of shape: math:`(1, N, H)`.
            encoder_state: tensor of shape: math:`(L_{in}, N, H)`.
            target_sequences: tensor of shape: math:`(L_{out}, N)`.
            target_masks: tensor of shape: math:`(L_{out}, N)`.
            target_max_length: an integer number.

        Returns: loss
            * **loss**: tensor of shape: math`()`.
        """
        scheduled_sampling = (
            True if random.random() > self.sampling_probability else False
        )
        loss = torch.tensor(data=0, dtype=torch.float32).to(self.device)
        if not scheduled_sampling:
            for t in range(target_max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_state
                )
                decoder_input = target_sequences[t].view(1, -1)

                # Calculate and accumulate loss
                mask_loss = self.loss_fn(
                    y_pred=decoder_output,
                    y_true=target_sequences[t],
                    mask=target_masks[t],
                )
                loss += mask_loss

        else:
            for t in range(target_max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_state
                )

                decoder_input = torch.argmax(decoder_output, dim=1).unsqueeze(dim=0)

                # decoder_input = (
                #     torch.multinomial(
                #         input=torch.softmax(decoder_output, dim=1),
                #         num_samples=1,
                #     )
                #     .transpose(0, 1)
                #     .to(self.device)
                # )

                # Calculate and accumulate loss
                mask_loss = self.loss_fn(
                    y_pred=decoder_output,
                    y_true=target_sequences[t],
                    mask=target_masks[t],
                )
                loss += mask_loss

        return loss

    @staticmethod
    def generate_batches(
        dataloader: torch.utils.data.DataLoader,
    ) -> Iterator[Dict[str, torch.Tensor | int]]:
        """Creates a batch generator from a `DataLoader` that
        yields data dictionaries upon iteration.

        Args:
            dataloader (torch.utils.data.DataLoader): The input
                dataloader.

        Returns:
            Iterator[Dict[str, torch.Tensor]]: An iterator
                over data dictionaries.
        """
        for data_dict in dataloader:
            input_sequences = data_dict["input_sequence"]
            input_lengths = data_dict["input_length"].numpy()
            target_sequences = data_dict["target_sequence"]
            target_lengths = data_dict["target_length"]
            target_masks = data_dict["target_mask"]
            sorted_length_indices = input_lengths.argsort()[::-1].tolist()
            yield {
                "input_sequences": input_sequences[sorted_length_indices].transpose(
                    0, 1
                ),
                "input_lengths": data_dict["input_length"][sorted_length_indices],
                "target_sequences": target_sequences[sorted_length_indices].transpose(
                    0, 1
                ),
                "target_max_length": target_lengths[sorted_length_indices].max().item(),
                "target_masks": target_masks[sorted_length_indices].transpose(0, 1),
            }

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        tqdm_bar: tqdm,
    ) -> None:
        """Trains and validates a sequence-to-sequence text generation model
        consisting of encoder and decoder layers.

        The training process includes:
            - Tracking training progress with MLFlow and custom tqdm bar
            - Checkpointing
            - Optional learning rate reduction
            - Optional early stopping

        If :attr:`enable_early_stop` is ``False`` a checkpoint is saved after
        every epoch.

        Args:
            train_dataloader (torch.utils.data.DataLoader): Dataloader for
                creating the training batch generator.
            val_dataloader (torch.utils.data.DataLoader): Dataloader for
                creating the validation batch generator.
            tqdm_bar (tqdm): Custom tqdm bar for the training.
        """
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        for epoch_index in range(self.last_epoch + 1, self.epochs + 1):
            self.sampling_probability = inverse_sigmoid_decay(
                decay=self.sampling_decay, epoch=epoch_index
            )

            train_loss = self._train_step(
                dataloader=train_dataloader,
                tqdm_bar=tqdm_bar,
                epoch_index=epoch_index,
            )
            val_loss = self._val_step(
                dataloader=val_dataloader,
                tqdm_bar=tqdm_bar,
                epoch_index=epoch_index,
            )

            print("\n")
            logger.info(
                f"===>>> epoch: {epoch_index} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"sampling_prob: {self.sampling_probability:.4f}"
            )

            mlflow.log_metrics(
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                step=epoch_index + self.last_epoch,
            )

            if self.enable_early_stop and self.early_stopper:
                self.early_stopper(
                    epoch=epoch_index + self.last_epoch,
                    embedding=self.embedding,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    encoder_optimizer=self.encoder_optimizer,
                    decoder_optimizer=self.decoder_optimizer,
                    val_loss=val_loss,
                )
            else:
                write_checkpoint(
                    epoch=epoch_index + self.last_epoch,
                    embedding=self.embedding,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    encoder_optimizer=self.encoder_optimizer,
                    decoder_optimizer=self.decoder_optimizer,
                    val_loss=val_loss,
                    path=self.checkpoint_path,
                )

            if self.scheduler:
                self.scheduler.step(val_loss)

            if (
                self.enable_early_stop
                and self.early_stopper
                and self.early_stopper.early_stop
            ):
                logger.info("Stopping training process due to early stopping.")
                break
            else:
                tqdm_bar.update()
                continue

    def _train_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        tqdm_bar: tqdm,
        epoch_index: int,
    ) -> float:
        """Trains a sequence-to-sequence text generation model for a
        single epoch.

        Turns the encoder and decoder to `train` mode and then runs
        through  all the required training steps (forward pass, loss
        calculation, optimizer step).

        Args:
            dataloader (torch.utils.data.DataLoader): The training
                dataloader for creating the training batch generator.
            tqdm_bar (tqdm): Custom tqdm bar for the training process.
            epoch_index (int): Current epoch.

        Returns:
          float: The training loss.
        """
        batch_generator = self.generate_batches(dataloader)

        self.encoder.train()
        self.decoder.train()

        running_loss = 0
        for batch_idx, data_dict in enumerate(BackgroundGenerator(batch_generator)):
            desc = (
                f"Training: [{epoch_index}/{self.epochs}] | "
                f"[{batch_idx}/{len(dataloader)}]"
            )
            tqdm_bar.set_description(desc=desc)
            tqdm_bar.set_postfix({"loss": running_loss})

            data_dict = dict(data_dict)
            input_sequences = data_dict["input_sequences"].to(self.device)
            input_lengths = data_dict["input_lengths"].to("cpu")
            target_sequences = data_dict["target_sequences"].to(self.device)
            target_max_length = data_dict["target_max_length"]
            target_masks = data_dict["target_masks"].to(self.device)
            batch_size = input_sequences.size(1)

            encoder_state, encoder_hidden = self.encoder(input_sequences, input_lengths)

            # Create initial decoder input starting with an SOS token
            # for each sentence
            decoder_input = torch.LongTensor(
                [[self.vocab.start_seq_index for _ in range(batch_size)]]
            ).to(self.device)

            # Set initial decoder hidden state to the encoder's final
            # hidden state
            decoder_hidden = encoder_hidden[: self.decoder.num_layers]

            loss = self.compute_metrics(
                decoder_input=decoder_input,
                decoder_hidden=decoder_hidden,
                encoder_state=encoder_state,
                target_sequences=target_sequences,
                target_masks=target_masks,
                target_max_length=target_max_length,
            )

            self.encoder.zero_grad()
            self.decoder.zero_grad()
            loss.backward()

            if self.clip_factor is not None:
                _ = nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.clip_factor
                )
                _ = nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), self.clip_factor
                )

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            running_loss += ((loss.item() / target_max_length) - running_loss) / (
                batch_idx + 1
            )

        return running_loss

    def _val_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        tqdm_bar: tqdm,
        epoch_index: int,
    ) -> float:
        """Validates a sequence-to-sequence text generation model for a
        single epoch.

        Turns the encoder and decoder layers to `eval` mode and performs
        a forward pass on the validation data.

        Args:
            dataloader (torch.utils.data.DataLoader): The validation
                dataloader for creating the validation batch generator.
            tqdm_bar (tqdm): Tqdm bar for tracking the validation process.
            epoch_index (int): Current epoch.

        Returns:
          float: The validation loss.
        """
        batch_generator = self.generate_batches(dataloader)

        self.encoder.eval()
        self.decoder.eval()

        running_loss = 0
        with torch.inference_mode():
            for batch_idx, data_dict in enumerate(BackgroundGenerator(batch_generator)):
                desc = (
                    f"Validation: [{epoch_index}/{self.epochs}] | "
                    f"[{batch_idx}/{len(dataloader)}]"
                )
                tqdm_bar.set_description(desc=desc)
                tqdm_bar.set_postfix({"loss": running_loss})

                data_dict = dict(data_dict)
                input_sequences = data_dict["input_sequences"].to(self.device)
                input_lengths = data_dict["input_lengths"].to("cpu")
                target_sequences = data_dict["target_sequences"].to(self.device)
                target_max_length = data_dict["target_max_length"]
                target_masks = data_dict["target_masks"].to(self.device)
                batch_size = input_sequences.size(1)

                encoder_state, encoder_hidden = self.encoder(
                    input_sequences, input_lengths
                )

                # Create initial decoder input starting with an SOS token
                # for each sentence
                decoder_input = torch.LongTensor(
                    [[self.vocab.start_seq_index for _ in range(batch_size)]]
                ).to(self.device)

                # Set initial decoder hidden state to the encoder's final
                # hidden state
                decoder_hidden = encoder_hidden[: self.decoder.num_layers]

                # Decoder forward pass and loss calculation
                loss = self.compute_metrics(
                    decoder_input=decoder_input,
                    decoder_hidden=decoder_hidden,
                    encoder_state=encoder_state,
                    target_sequences=target_sequences,
                    target_masks=target_masks,
                    target_max_length=target_max_length,
                )

                running_loss += ((loss.item() / target_max_length) - running_loss) / (
                    batch_idx + 1
                )

        return running_loss
