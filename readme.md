# 305B Final Project
Title: Large Language Model with Self-Corrector

Author: Peter Park

This repository provides replication codes for the HW4 Part 2 mini project. The project adds `corrector` on top of the vanilla transformer given in the homework. The codes consist of three parts.

## main.py
Executes functions and classes.

## preprocess.py
Implements text preprocessing and helper functions.

Available functions: `encode`, `decode`, `get_batch`, `get_corrector_batch`, `estimate_loss`, `estimate_corrector_loss`

`get_corrector_batch` and `estimate_corrector_loss` are the adaption of helper functions to `corrector`.

## architecture.py
Implements transformer and neural network architectures.

Available classes: `MultiHeadAttention`, `FeedForward`, `DeepFeedForward`, `TransformerBlock`, `Predictor`, `Corrector`, `Generator`

`DeepFeedForward`: neural network with two nonlinear layers

`Corrector`: blend the baseline transformer with `DeepFeedForward`

`Generator`: takes `Predictor` and `Corrector` as arguments and repeats simulate-pick.

## debuggers.py
Collects debugger functions.
