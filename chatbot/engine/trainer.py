import random
from collections.abc import Iterator
from typing import Dict, Optional

import mlflow
import torch
import torch.utils.data
from prefetch_generator import BackgroundGenerator
from torch import nn
from tqdm import tqdm

from chatbot import logger

# from chatbot.engine.evaluator import Evaluator, GreedySearchDecoder
from chatbot.engine.utils import EarlyStopping, load_general_checkpoint
from chatbot.utils.data.vectorizer import SequenceVectorizer


class Trainer:
    """Class for training a seq-to-seq model with PyTorch.

    The model consists of encoder and decoder layers, each trained
    separately using its own optimizer.

    Incorporates functionalities such as early stopping, resume from
    checkpoint, and MLFlow tracking.

    Args:
        checkpoint_path (str): The file path to save or load a checkpoint.
        encoder (torch.nn.Module): The encoder layer.
        decoder (torch.nn.Module): The decoder layer.
        encoder_optimizer (torch.optim.Optimizer): The optimizer for updating
            the parameters of the encoder.
        decoder_optimizer (torch.optim.Optimizer): The optimizer for updating
            the parameters of the decoder.
        vectorizer (CharacterVectorizer): Vectorizes a text sequence
            to observations `X` and targets `y`.
        epochs (int, optional): Number of training epochs
            (default=5).
        patience (int, optional): Number of epochs to wait before
        early stopping (default=5).
        delta (float, optional): Minimum change in monitored quantity
            to qualify as an improvement (default=0).
        clip_factor (float, optional): Max norm value for gradient clipping
            Defaults to ``None``.
        teacher_forcing_ratio (float, optional):
        resume (bool, optional): If ``True``, resumes training from
            the specified checkpoint. Defaults to ``False``.
        scheduler (optional): Reduces the learning rate when the
            validation loss stops improving. Defaults to ``None``.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        vectorizer: SequenceVectorizer,
        checkpoint_path: str,
        epochs: int = 5,
        patience: int = 5,
        delta: float = 0.0,
        clip_factor: Optional[float | int] = None,
        teacher_forcing_ratio: float = 0.0,
        resume: bool = False,
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.scheduler = scheduler
        self.vectorizer = vectorizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.resume = resume
        self.clip_factor = clip_factor
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.early_stopping: EarlyStopping = EarlyStopping(
            patience=patience, delta=delta, path=checkpoint_path, verbose=True
        )
        # self.evaluator = Evaluator(vectorizer=vectorizer)

    def mask_nll_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        crossEntropy = -torch.log(
            torch.gather(input=y_pred, dim=1, index=y_true.view(-1, 1)).squeeze(1)
        )
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss

    def compute_loss(
        self,
        decoder_input: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_state: torch.Tensor,
        target_sequences: torch.Tensor,
        target_masks: torch.Tensor,
        target_max_length: int,
    ) -> torch.Tensor:
        batch_size = target_sequences.size(1)

        use_teacher_forcing = (
            True if random.random() < self.teacher_forcing_ratio else False
        )

        loss = torch.tensor(data=0, dtype=torch.float32).to(self.device)
        if use_teacher_forcing:
            for t in range(target_max_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_state
                )
                decoder_input = target_sequences[t].view(1, -1)

                # Calculate and accumulate loss
                mask_loss = self.mask_nll_loss(
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

                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(k=1, dim=1)
                decoder_input = torch.LongTensor(
                    [[topi[i][0] for i in range(batch_size)]]
                ).to(self.device)

                # Calculate and accumulate loss
                mask_loss = self.mask_nll_loss(
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
        """Trains a seq-to-seq text generation PyTorch encoder.

        The training routine expects from the dataloaders to provide
        two sequence of integers which represent the token observations,
        and the token targets at each time step.

        Performs the training using the provided dataloaders, loss function,
        and optimizer. It also performs evaluation on the validation data at
        the end of each epoch. Checkpointing is supported, optionally allowing
        for the resumption of training from a saved checkpoint.

        The training process includes learning rate reduction and early stopping
        to prevent over-fitting. The training loop stops if the validation loss
        does not improve for a certain number of epochs, defined from the
        :attr:`patience` class attribute.

        Args:
            train_dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                training data.
            val_dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for providing batches of
                validation data.
            tqdm_bar (tqdm): Custom tqdm bar for the training.
        """
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        last_epoch = 0
        if self.resume:
            checkpoint = load_general_checkpoint(
                encoder=self.encoder,
                decoder=self.decoder,
                encoder_optimizer=self.encoder_optimizer,
                decoder_optimizer=self.decoder_optimizer,
                filepath=self.checkpoint_path,
            )
            self.encoder = checkpoint["encoder"].to(self.device)
            self.decoder = checkpoint["decoder"].to(self.device)
            self.encoder_optimizer = checkpoint["encoder_optimizer"]
            self.decoder_optimizer = checkpoint["decoder_optimizer"]
            loss_value = checkpoint["val_loss"]
            last_epoch = checkpoint["epoch"]
            # self.epochs -= last_epoch
            # tqdm_bar = tqdm(
            #     desc="Training routine",
            #     total=self.epochs,
            #     position=0,
            # )
            logger.info(
                f"Resume training from general checkpoint: {self.checkpoint_path}."
            )
            logger.info(f"Last training loss value: {loss_value:.4f}")
            logger.info(
                f"Resuming from epoch {last_epoch + 1}. "
                f"Remaining epochs: {self.epochs}"
            )

        for epoch_index in range(0, self.epochs):
            train_loss = self._train_step(
                dataloader=train_dataloader,
                tqdm_bar=tqdm_bar,
                epoch_index=epoch_index + last_epoch,
            )
            val_loss = self._val_step(
                dataloader=val_dataloader,
                tqdm_bar=tqdm_bar,
                epoch_index=epoch_index + last_epoch,
            )

            print("\n")
            logger.info(
                f"===>>> epoch: {epoch_index + 1 + last_epoch} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
            )

            mlflow.log_metrics(
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                step=epoch_index + 1 + last_epoch,
            )

            self.early_stopping(
                epoch=epoch_index + 1 + last_epoch,
                encoder=self.encoder,
                decoder=self.decoder,
                encoder_optimizer=self.encoder_optimizer,
                decoder_optimizer=self.decoder_optimizer,
                val_loss=val_loss,
            )

            if self.scheduler:
                self.scheduler.step(val_loss)

            if self.early_stopping.early_stop:
                logger.info("Training stopped due to early stopping.")
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
        """Trains a seq-to-seq text generation PyTorch encoder
        for a single epoch.

        Turns the target encoder to `train` mode and then runs
        through all the required training steps (forward pass,
        loss calculation, optimizer step).

        Args:
            dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for creating the training
                batch generator.
            tqdm_bar (tqdm): Custom tqdm bar for the training
                process.
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
                f"Training: [{epoch_index + 1}/{self.epochs}] | "
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

            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.LongTensor(
                [[self.vectorizer.vocab.begin_seq_index for _ in range(batch_size)]]
            ).to(self.device)

            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[: self.decoder.num_layers]

            loss = self.compute_loss(
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

            # plot_grad_flow(self.model.named_parameters())
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # Compute the running loss
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
        """Validates a seq-to-seq text generation PyTorch encoder
        for a single epoch.

        Turns the target encoder to `eval` encoder and then performs
        a forward pass on the validation data.

        Args:
            dataloader (torch.utils.data.DataLoader):
                A `DataLoader` instance for creating the validation
                batch generator.
            tqdm_bar (tqdm): Custom tqdm bar for the validation
                process.
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
                    f"Validation: [{epoch_index + 1}/{self.epochs}] | "
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

                # Create initial decoder input (start with SOS tokens for each sentence)
                decoder_input = torch.LongTensor(
                    [[self.vectorizer.vocab.begin_seq_index for _ in range(batch_size)]]
                ).to(self.device)

                # Set initial decoder hidden state to the encoder's final hidden state
                decoder_hidden = encoder_hidden[: self.decoder.num_layers]

                # Decoder forward pass and loss calculation
                loss = self.compute_loss(
                    decoder_input=decoder_input,
                    decoder_hidden=decoder_hidden,
                    encoder_state=encoder_state,
                    target_sequences=target_sequences,
                    target_masks=target_masks,
                    target_max_length=target_max_length,
                )

                # Compute the running loss
                running_loss += ((loss.item() / target_max_length) - running_loss) / (
                    batch_idx + 1
                )

        return running_loss
