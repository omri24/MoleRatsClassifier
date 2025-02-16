# MoleRatsClassifier
****
## General description

Blind mole rats (BMR) are mammals that live under-ground and communicate by ramming their heads against their tunnel walls.
Our goal is to distinguish between individual BMR, and hopefully to point-out properties of the audio that take part in the classification.
This project is done in cooperation with prof. Yossi Yovel from Tel-Aviv University. 

## The dataset
The dataset consist of 5 different BMR that communicate. The audio is given in labeled multi-channel WAV files.

## The classifier
The classifier consists of 2 parts:

1. WavLM encoder, description in this paper: https://arxiv.org/abs/2110.13900.
2. Classification over the embedding space of the encoder, a few methods were tested.

## Results

to be continued...