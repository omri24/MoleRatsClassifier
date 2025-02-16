# MoleRatsClassifier

## General description

Blind mole rats (BMR) are mammals that live under-ground and communicate by ramming their heads against their tunnel walls.
Our goal is to distinguish between individual BMR, and hopefully to point-out properties of the audio that take part in the classification.
This project is done in cooperation with prof. Yossi Yovel from Tel-Aviv University. 

## The dataset
The dataset consist of 5 different BMR that communicate. The audio is given in labeled multi-channel WAV files.

## Analyzing the data
The plots below show a pulse created by a BMR, presented in time domain and in frequency domain (FFT and STFT). 

<img src="BMR in domains.png" alt="Plot Example" width="1000" height="1000">

The plots below show typical channel-noise, presented in time domain and in frequency domain (FFT and STFT). 

<img src="noise in domains.png" alt="Plot Example" width="1000" height="1000">

Out of the three representations of the data shown above, the clearest representation of the pulse is the time domain representation.
Therefore, classification in time domain was chosen over classification in frequency domain.

## The classifier
The classifier consists of 2 parts:

1. WavLM encoder, description in this paper: https://arxiv.org/abs/2110.13900, and in HuggingFace: https://huggingface.co/microsoft/wavlm-base-plus-sv.
2. Classification over the embedding space of the encoder, a few methods were tested.

Over all, the classification is between 5 classes of BMR and one class of noise (6 classes in total).

## Noise handling
Initially, the recordings were sampled uniformly, to get a notion how significant is the noise. 
Classification between samples from different recordings reached accuracy of about 90%. 
Conclusion is that the background noise is significant, and to get accurate classification, 
it is necessary to distinguish also between noise and BMR, not only between different BMR individuals.

## Results

| Classification method | Accuracy, range: [0-1] | Training time, HW: NVIDIA GeForce MX450 [sec] |
|-----------------------|------------------------|-----------------------------------------------|
| K-means | 0.0612 | 0.2118 |
| KNN (k = 3) | 0.7146 | 9.0969 |
| SVM (rbf kernel) | 0.7353 | 15.4334 |
| FC-NN | 0.7735 | 5.3698 |
| FC-NN | 0.7673 | 21.4524 |
| FC-NN | 0.7663 | 21.0988 |
| LSTM | 0.7715 | 23.26 |

## The embedding space

The plot shows samples containing pulses created by BMR, and samples containing only channel noise.  

<img src="Embedding_space_plot.png" alt="Plot Example" width="1000" height="500">

Some trends may be seen, however, it is not clear what makes the classification possible.

## Conclusions

Empirically, classification between different BMR individuals and between noise is demonstrated.
However, the properties of the audio that allow the classification are not clear.