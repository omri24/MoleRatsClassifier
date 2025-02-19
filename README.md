# MoleRatsClassifier

## General description

Blind mole rats (BMR) are mammals that live in tunnels under-ground 
and communicate by ramming their heads against the tunnel walls.

This project has 2 goals:  
1. Distinguishing between sounds produced by different individual BMR.
2. Pointing out audio properties that take part in the classification.

The data used in the project was provided by prof. Yossi Yovel from Tel-Aviv University. 

## The dataset

The dataset contains recordings of 5 different BMR. 
The audio is given in labeled multi-channel .WAV files.

## Analyzing the data

The plots below show a pulse created by a BMR, 
presented in time domain and in frequency domain (FFT and STFT). 

<img src="BMR in domains.png" alt="Plot Example" width="1000" height="1000">

The plots below show typical channel-noise, presented in time domain and in frequency domain (FFT and STFT). 

<img src="noise in domains.png" alt="Plot Example" width="1000" height="1000">

Out of the 3 representations of the data, the clearest representation of the pulse is the time domain representation.
Therefore, classification in time domain was chosen over classification in frequency domain.

## The classifier

The classifier consists of 2 parts:

1. WavLM encoder, suitable for speech recognition and for speaker verification tasks. Description in the paper: 
https://arxiv.org/abs/2110.13900 and in HuggingFace: https://huggingface.co/microsoft/wavlm-base-plus-sv.
2. Classifier over the embedding space of the encoder, a few methods were tested.

Over all, the classification is between 5 classes of BMR and one class of noise (6 classes in total).

## Noise handling

Initially, the recordings were sampled uniformly, to get a notion for how significant is the noise. 
Classification between samples from different recordings reached accuracy of about 80%. 

The conclusion is that the background noise is significant, and in order to get a valid classifier, 
it is necessary to distinguish also between noise and recorded BMR, not only between pulses created by different BMR.

## Results

| Classification method | Accuracy, range: [0-1] |
|-----------------------|------------------------|
| K-means | 0.0612 |
| KNN, k=3, cosine distance | 0.7146 |
| SVM, rbf kernel | 0.7353 |
| FC-NN, 1 hidden layer | 0.7725 |
| FC-NN, 2 hidden layers, drop-out | 0.7777 |
| FC-NN, 3 hidden layers, drop-out, batch-norm | 0.7425 |
| LSTM | 0.7642 |

Over all, maximal accuracy achieved is around 77%.

For all embedding space classification methods, training time was less than 30 seconds on NVIDIA GeForce MX450 GPU.
## The embedding space

The plot below shows samples containing pulses created by BMR, and samples containing only channel noise.  

<img src="Embedding_space_plot.png" alt="Plot Example" width="1000" height="500">

Some trends may be pointed out, however, it is not clear what makes the classification possible.

## Conclusions

1. Empirically, classification between different BMR individuals and between noise is demonstrated.


2. Almost all classification methods over the embedding space resulted in similar accuracy rates, 
which implies that the encoder is the significant part of the classification, not the classifiers over the embedding space. 


3. Conclusion from #2 - the WavLM encoder used, that was trained for speech recognition and speaker verification tasks on human language, 
preforms well in the equivalent tasks for BMR 'language'.


4. Properties of the audio that are significant for the classification process are not clear, 
as could be expected. Deep learning algorithms are in that sense
'black-boxes' - it's almost impossible to point out why they decide one way or another. 


5. The channel-noise is probably not a significant audio property for the classification process.
If it was, accuracy rates in confusion matrix 3 (between noise and BMR pulses) would probably be lower. 
   
## Appendix: confusion matrices

The matrices below relate to the FC-NN with 3 hidden layers.

<img src="cmat2.png" alt="Plot Example" width="1000" height="600">

<img src="cmat4.png" alt="Plot Example" width="1000" height="600">

<img src="cmat6.png" alt="Plot Example" width="1000" height="600">
