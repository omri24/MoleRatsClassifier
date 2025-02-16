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


| Classification method | Accuracy | Training time on NVIDIA GeForce MX450 [sec] |
|-----------------------|----------|---------------------------------------------|
| K-means | 0.0612 | 0.2224 |
| KNN (k = 3) | 0.7146 | 9.1354 |
| SVM (rbf kernel) | 0.7353 | 1.5177 |
| FC-NN | 0.7735 | 5.3726 |
| FC-NN | 0.7715 | 21.516 |
| FC-NN | 0.7611 | 21.1105 |
| LSTM | 0.7756 | 23.0618 |