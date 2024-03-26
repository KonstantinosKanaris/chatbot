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
