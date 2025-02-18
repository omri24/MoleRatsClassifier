# MoleRatsClassifier

## General description

Blind mole rats (BMR) are mammals that live under-ground and communicate by ramming their heads against their tunnel walls.
The goal is to distinguish between individual BMR, and hopefully to point-out properties of the audio that take part in the classification.
The data used in the project was provided by prof. Yossi Yovel from Tel-Aviv University. 

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

1. WavLM encoder, suitable for speech recognition. Description in this paper: https://arxiv.org/abs/2110.13900, and in HuggingFace: https://huggingface.co/microsoft/wavlm-base-plus-sv.
2. Classification over the embedding space of the encoder, a few methods were tested.

Over all, the classification is between 5 classes of BMR and one class of noise (6 classes in total).

## Noise handling
Initially, the recordings were sampled uniformly, to get a notion how significant is the noise. 
Classification between samples from different recordings reached accuracy of about 90%. 
Conclusion is that the background noise is significant, and to get accurate classification, 
it is necessary to distinguish also between noise and BMR, not only between different BMR individuals.

## Results

| Classification method (over embedding space) | Accuracy<br/> Range: [0-1] | Training time<br/> HW: NVIDIA GeForce MX450 [sec] |
|----------------------------------------------|----------------------------|----------------------------------------------|
| K-means                                      | 0.0612                     | 0.3749 |
| KNN, k=3, cosine distance                    | 0.7146                     | 10.157 |
| SVM, rbf kernel                              | 0.7353                     | 1.6959 |
| FC-NN, 1 hedden layer                        | 0.7777                     | 5.3749 |
| FC-NN, 2 hidden layers, drop-out             | 0.7797                     | 21.5029 |
| FC-NN, 3 hidden layers, drop-out, batch-norm | 0.7746                     | 21.0224 |
| LSTM, using last hidden state                | 0.7797                     | 22.9475 |

Over all, maximal accuracy achieved is around 77%.

## The embedding space

The plot shows samples containing pulses created by BMR, and samples containing only channel noise.  

<img src="Embedding_space_plot.png" alt="Plot Example" width="1000" height="500">

Some trends may be seen, however, it is not clear what makes the classification possible.

## Conclusions

1. Empirically, classification between different BMR individuals and between noise is demonstrated.
2. The properties of the audio that allow the classification are not clear.
3. Almost all classification methods over the embedding space resulted in similar accuracy.
4. Conclusion from #3 - the WavLM encoder used, that was trained for speech recognition and speaker verification tasks on human language, 
preforms well in the equivalent tasks for BMR 'language'.