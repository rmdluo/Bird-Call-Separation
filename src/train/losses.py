import torch
from torchaudio.transforms import Spectrogram
from itertools import permutations


class SISNRLoss(torch.nn.Module):
    def __init__(self, loss_clipping=30, eps=1e-5):
        super().__init__()
        self.loss_clipping = loss_clipping
        self.eps = eps

    def _calculate_pair_sisnr(self, prediction_waveforms, target_waveforms, pairing):
        total = 0
        for i in range(len(pairing)):
            j = pairing[i]

            prediction = prediction_waveforms[i]
            prediction = prediction - prediction.mean()
            target = target_waveforms[j]
            target = target - target.mean()

            projection = (
                (prediction * target).sum() / ((target * target).sum() + self.eps)
            ) * target

            sisnr = 20 * torch.log10(
                (projection.norm() + self.eps)
                / ((projection - prediction).norm() + self.eps)
            )

            if self.loss_clipping is not None:
                total += torch.clamp(sisnr, max=self.loss_clipping)
            else:
                total += sisnr
        return total

    def forward(self, prediction_waveforms, target_waveforms):
        assert prediction_waveforms.shape == target_waveforms.shape
        num_waveforms = prediction_waveforms.shape[0]
        combinations = list(permutations(range(num_waveforms)))

        scores = [
            self._calculate_pair_sisnr(
                prediction_waveforms, target_waveforms, combination
            )
            for combination in combinations
        ]

        scores = torch.stack(scores)
        best_combo = torch.argmax(scores).item()
        return (
            -scores[best_combo],
            combinations[best_combo],
        )


class STFTSISNRLoss(torch.nn.Module):
    def __init__(
        self, loss_clipping=30, eps=1e-5, n_fft=256, win_length=256, hop_length=128
    ):
        super().__init__()
        self.loss_clipping = loss_clipping
        self.eps = eps
        self.stft = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
        )

    def _calculate_pair_sisnr(self, prediction_waveforms, target_waveforms, pairing):
        total = 0
        for i in range(len(pairing)):
            j = pairing[i]

            prediction = prediction_waveforms[i]
            prediction = prediction - prediction.mean()
            target = target_waveforms[j]
            target = target - target.mean()

            projection = (
                (prediction * target).sum() / ((target * target).sum() + self.eps)
            ) * target

            sisnr = 20 * torch.log10(
                (self.stft(projection).abs().norm() + self.eps)
                / (
                    (self.stft(projection).abs() - self.stft(prediction).abs()).norm()
                    + self.eps
                )
            )

            total += torch.clamp(sisnr, max=self.loss_clipping)
        return total

    def forward(self, prediction_waveforms, target_waveforms, combination=None):
        assert prediction_waveforms.shape == target_waveforms.shape

        if combination is None:
            num_waveforms = prediction_waveforms.shape[0]
            combinations = list(permutations(range(num_waveforms)))

            scores = [
                self._calculate_pair_sisnr(
                    prediction_waveforms, target_waveforms, combination
                )
                for combination in combinations
            ]

            return -torch.stack(scores).max()
        else:
            return -self._calculate_pair_sisnr(
                prediction_waveforms, target_waveforms, combination
            )
