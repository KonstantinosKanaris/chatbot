# Character-based Text Generation with PyTorch

## Table of Contents
* [Overview](#Overview)
* [Key Features](#Key--Features)
* [Data](#Data)
* [Development](#Development)
* [Execute](#Execute)
* [Experiment Tracking](#Experiment--Tracking)
* [References](#References)

## Overview 🔍
A chatbot framework with PyTorch for training and evaluating seq-to-seq models.

Implements 2 high-level processes:
- Training of a seq-to-seq PyTorch model consisting of encoder and decoder layers
- Chat with the trained bot (evaluation)

Both processes are fully configurable through dedicated configuration files. For
instance, for the training process, users can define training hyperparameters, such
as number of epochs, batch size, model/optimizer settings, in a YAML file. This allows for
easy customization and experimentation with different configurations without modifying
the source code.

### Project Structure 🌲
```
chatbot
├── .gitignore
├── .pre-commit-config.yaml             | Pre-commit hooks
├── Makefile                            | Development commands for code formatting, linting, etc
├── Pipfile                             | Project's dependencies and their versions using pipenv format
├── Pipfile.lock                        | Auto-generated file that locks the dependencies to specific versions for reproducibility
├── README.md
├── chatbot                             | The main Python package containing the project's source code
│   ├── __about__.py              | Metadata about the project, i.e., version number, author information, etc
│   ├── __init__.py
│   ├── __main__.py               | The entry point for running the package as a script. Calls one of the controllers
│   ├── controllers.py            | Contains the training controller
│   ├── engine
│   │   ├── __init__.py
│   │   ├── evaluator.py    | Contains the evaluation process
│   │   ├── trainer.py      | Contains the training process
│   │   └── utils.py        | Auxilliary functions/classes for training and evaluation such as EarlyStopping
│   ├── models
│   │   ├── __init__.py
│   │   ├── attention.py    | Luong Attention layer for the seq2seq model
│   │   ├── decoders.py     | Decoder with Luong Attention and decoders/samplers for inference
│   │   ├── embeddings.py   | Creates embedding layer with or without pretrained embeddings
│   │   ├── encoders.py     | Encoder for the seq2seq model
│   │   └── loss.py         | Implementation of custom loss methods, i.e., MaskedNLLLoss
│   └── utils
│       ├── __init__.py
│       ├── aux.py                | Auxilliary functions/classes used across the project
│       └── data
│           ├── __init__.py
│           ├── datasets.py       | Cornell Dialogs Dataset
│           ├── vectorizer.py     | Vectorizes text sequences to numbers. Token-based vectorization
│           └── vocabulary.py     | Maps tokens to indices and vice versa

├── checkpoints                         | Checkpoints directory
│   └── ...
├── colours.mk                          | A Makefile fragment containing color codes for terminal output styling
├── configs
│   ├── evaluation.yaml           | Configuration parameters for the evaluation process
│   ├── experiments.yaml          | Configuration parameters for the training process
│   └── logging.ini               | Configuration file for Python's logging module
├── data_preparation.ipynb              | Cornel dialogs data downloading and preparation
├── docs                                | Sphinx generated documentation
│   ├──build
│   ├──source
│   └── ...
└── mypy.ini                            | Configuration file for the MyPy static type checker
```

## Key Features 🔑

### General
* **Customizable Experiments**: Easy configuration of training experiments by defining the necessary
training parameters in a yaml file.
* **Experiment Tracking**: Use of MLFlow for tracking training experiments
* **Available Documentation**: [Documentation](docs/build/html/modules.html)

### Training Process
* **Custom TQDM Bar**: ```Training: [11/50] | [155/156]:   22%|███▍       , loss=2.44 [01:40<21:29]```
* **Checkpointing**: Option to resume training process from the last saved checkpoint
* **EarlyStopping**: Training process stops when the model's performance stops improving on the validation set
* **LRReduction**: Learning rate reduction during training (Not implemented yet)

### Seq2Seq Model
* **Encoder-Decoder Architecture**: Encoder and decoder layers are trained with separate weights, optimizers and
lr schedulers
* **Embedding Layer**: Can be initialized with pre-trained word embeddings, i.e. GLoVe embeddings
(Implementation exists)
* **Luong Attention**: Implementation of a decoder with global Luong attention based on the paper:
https://arxiv.org/abs/1508.04025
* **Scheduled Sampling**: Implementation of Scheduled Sampling with custom Decay Schedulers based on the paper:
https://arxiv.org/abs/1506.03099

### Evaluation
* **Various Search Decoders**: Greedy Search Decoder, Random (multinomial) Search Decoder,
(Beam Search Decoder-> TODO)
* **ChatBot UI**: Not implemented yet

## Data 📄

## Development 🐍
Clone the repository:
  ```bash
  $ git clone https://github.com/KonstantinosKanaris/chatbot.git
  ```

### Set up the environment

#### Create environment
Python 3.10 is required.

- Create the environment and install the dependencies:
    ```bash
    $ pipenv --python 3.10
    $ pipenv install --dev
    ```
- Enable the newly-created virtual environment, with:
    ```bash
    $ pipenv shell
    ```

## Execute 🚀

### Training
> Define training parameters in the configuration file and from the project's root directory execute:
>```bash
> $ python -m chatbot train --config <path_to_training_yaml_file>
>```
> To resume training from a saved checkpoint execute:
>```bash
> $ python -m chatbot train --config ./configs/experiments.yaml --resume_from_checkpoint yes
>```
> The checkpoint path is defined in the configuration file.

### Evaluation
> Start chatting with the bot with:
>```bash
> $ python -m chatbot evaluate --config <path_to_evalation_yaml_file>
>```
> **ℹ️ Important**
>
> Make sure that the initialization parameters for each of the seq2seq model components defined in
> the evaluation configuration file match with those in the training configuration file.

## Experiment Tracking 📉
Not implemented yet.

## References 📚
1. [Effective Approaches to Attention-based Neural Machine Translation. Luong, M.-T., Pham, H., & Manning, C. D.(2015)](https://arxiv.org/abs/1508.04025)
2. [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer.](https://arxiv.org/abs/1506.03099)
3. [Perkins, Jacob. *Natural Language Processing*. O'Reilly Media, 2017.](https://www.oreilly.com/library/view/natural-language-processing/9781491978221/)
4. [Matthew Inkawhich, PyTorch Chatbot Tutorial. PyTorch Tutorials.](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
