import torch
import torchaudio
import torch.nn.functional as F


def change_sample_rate(waveform, sample_rate, target_sample_rate):
    if sample_rate == target_sample_rate:
        return waveform, sample_rate

    waveform = torchaudio.functional.resample(
        waveform,
        sample_rate,
        target_sample_rate,
    )
    sample_rate = target_sample_rate

    return waveform, sample_rate


def change_duration(waveform, sample_rate, duration):
    waveform = waveform[:, : (sample_rate * duration)]
    if waveform.shape[-1] < (sample_rate * duration):
        waveform = F.pad(
            waveform,
            (0, (sample_rate * duration) - waveform.shape[-1]),
        )

    return waveform, sample_rate


def mix_waveforms(waveforms):
    """
    Input: Tensor of shape [num_waveforms, channels, samples]
    Note: this function assumes that they are all the same duration and sample rate.
    """
    return waveforms.sum(dim=0)


def convert_to_mono(waveform):
    return waveform.mean(dim=0, keepdim=True)
