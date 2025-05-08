from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer, WavLMForXVector, Wav2Vec2Processor
import soundfile as sf
import re
import os
import numpy as np
import torch
import random

path_to_dataset = "dataset"
paths_lst = [os.path.join(path_to_dataset, item) for item in os.listdir(path_to_dataset)]

def add_noise(audio, noise_level=1):
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def time_shift(audio, shift_max=0.1, sr=16000):
    """Shift audio forward or backward."""
    shift = int(random.uniform(-shift_max, shift_max) * sr)
    return np.roll(audio, shift)

# Configuration
WavLM_mode = "Xvec"
normalize = True
augmentation = False   # 'n' -> add_noise, 't' -> time_shift,
augmentation_dict = {"n": "add_noise", "r": "apply_reverb", "t": "time_shift", "f": "bandpass_filter"}
discard_even_labels = True     # Noise labels are even, and sometimes we want to process only BMR samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for path in paths_lst:
  if re.search("label", path):
    list_dir = os.listdir(path)
    list_dir = [os.path.join(path, item) for item in list_dir]
    label_finder = re.search("label", path)
    full_label_finder = re.search("label\d*", path)
    label_as_int = int(path[label_finder.end(): full_label_finder.end()])
    if discard_even_labels and label_as_int % 2 == 0:
      continue
    channel_finder = re.search("ch", path)
    full_channel_finder = re.search("ch\d*", path)
    exp_name_finder = re.search("R\dS\d", path)
    exp_name = path[exp_name_finder.start(): exp_name_finder.end()]
    channel_as_int = int(path[channel_finder.end(): full_channel_finder.end()])
    audio = []
    for item in list_dir:
      if re.search("channel", item):   # If 'channel' not in the file name, it's not a chopped wav file and should be discarded
        audio_sample, sample_rate = sf.read(item)
        if augmentation:
          exec(f"audio_sample = {augmentation_dict[augmentation]}(audio_sample)")
        audio.append(audio_sample)


    if WavLM_mode == "classification":
      # uncomment next line to use this option (which you probably don't want to do)
      # model = WavLMForSequenceClassification.from_pretrained("microsoft/wavlm-base-plus-sv", ignore_mismatched_sizes=True)
      feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
      tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
      processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
      inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    if WavLM_mode == "Xvec":
      feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
      model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
      inputs = feature_extractor(audio, sampling_rate=sample_rate, padding=True, return_tensors="pt")

    # calculate embeddings
    with torch.no_grad():
      outputs = model(**inputs)
      embeddings = outputs.embeddings
      if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
      else:
        embeddings = embeddings.cpu()

    # Save the embeddings
    embeddings_arr = embeddings.numpy()
    if augmentation:
      np.savetxt(f"{exp_name}_embeddings_ch{channel_as_int}_augmentation_{augmentation_dict[augmentation]}_label{label_as_int}.csv", embeddings_arr, delimiter=",")
      print(f"exported file {exp_name}_embeddings_ch{channel_as_int}_augmentation_{augmentation_dict[augmentation]}_label{label_as_int}.csv")
    else:
      np.savetxt(f"{exp_name}_embeddings_ch{channel_as_int}_label{label_as_int}.csv", embeddings_arr, delimiter=",")
      print(f"exported file {exp_name}_embeddings_ch{channel_as_int}_label{label_as_int}.csv")
