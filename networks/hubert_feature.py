import torchaudio
import torch
import librosa
from transformers import HubertForCTC, Wav2Vec2Processor

def refactorWaveform(audio_waveform_sample_rate):

    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model()
    waveform, sample_rate = audio_waveform_sample_rate

    device = torch.device("cpu")
    waveform = waveform.to(device)

    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    features, _ = model.extract_features(waveform) # num_layers=1
    feature_array = features[11].squeeze()
    feature_array_1d = torch.mean(feature_array,0)

    return feature_array_1d
    