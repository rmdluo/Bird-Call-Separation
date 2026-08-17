import torch
from einops import rearrange
from dataclasses import dataclass, field

import torchaudio

from models.common import RoPESelfAttention, SelfAttention, SwiGLU

from einops._torch_specific import allow_ops_in_compiled_graph

allow_ops_in_compiled_graph()


@dataclass
class ModifiedSRCorrNetEncoderBlockParameters:
    dim: int = 128
    nheads: int = 4
    dim_feedforward: int = 384
    kernel_size_feedforward: int = 7


@dataclass
class ModifiedSRCorrNetDecoderBlockParameters:
    dim: int = 128
    nheads: int = 4
    dim_feedforward: int = 384
    kernel_size_feedforward: int = 7


@dataclass
class ModifiedSRCorrNetParameters:
    n_fft: int = 256
    win_length: int = 256
    hop_length: int = 128
    num_encoder_blocks: int = 4
    num_decoder_blocks: int = 4
    num_speakers: int = 3
    dim: int = 128
    encoder_block_parameters: ModifiedSRCorrNetEncoderBlockParameters = field(
        default_factory=ModifiedSRCorrNetEncoderBlockParameters
    )

    decoder_block_parameters: ModifiedSRCorrNetDecoderBlockParameters = field(
        default_factory=ModifiedSRCorrNetDecoderBlockParameters
    )


class ModifiedSRCorrNetEncoderBlock(torch.nn.Module):
    def __init__(self, params: ModifiedSRCorrNetEncoderBlockParameters):
        super().__init__()

        self.freq_norm = torch.nn.LayerNorm(params.dim)
        self.freq_proj = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.dim,
                2 * params.dim_feedforward,
                params.kernel_size_feedforward,
                padding="same",
            ),
            SwiGLU(),
            torch.nn.Conv1d(
                params.dim_feedforward,
                params.dim,
                params.kernel_size_feedforward,
                padding="same",
            ),
        )
        self.freq_mhsa = torch.nn.Sequential(
            torch.nn.LayerNorm(params.dim), RoPESelfAttention(params.dim, params.nheads)
        )
        self.freq_norm_2 = torch.nn.LayerNorm(params.dim)
        self.freq_ffn = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.dim,
                2 * params.dim_feedforward,
                params.kernel_size_feedforward,
                padding="same",
            ),
            SwiGLU(),
            torch.nn.Conv1d(
                params.dim_feedforward,
                params.dim,
                params.kernel_size_feedforward,
                padding="same",
            ),
        )

        self.time_norm = torch.nn.LayerNorm(params.dim)
        self.time_proj = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.dim,
                2 * params.dim_feedforward,
                params.kernel_size_feedforward,
                padding="same",
            ),
            SwiGLU(),
            torch.nn.Conv1d(
                params.dim_feedforward,
                params.dim,
                params.kernel_size_feedforward,
                padding="same",
            ),
        )
        self.time_mhsa = torch.nn.Sequential(
            torch.nn.LayerNorm(params.dim), RoPESelfAttention(params.dim, params.nheads)
        )
        self.time_norm_2 = torch.nn.LayerNorm(params.dim)
        self.time_ffn = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.dim,
                2 * params.dim_feedforward,
                params.kernel_size_feedforward,
                padding="same",
            ),
            SwiGLU(),
            torch.nn.Conv1d(
                params.dim_feedforward,
                params.dim,
                params.kernel_size_feedforward,
                padding="same",
            ),
        )

    def forward(self, x):
        # Assume x is B x T x F x C
        time_batched = rearrange(x, "b t f c -> (b t) f c")

        freq_output_proj = self.freq_proj(
            self.freq_norm(time_batched).transpose(1, 2)
        ).transpose(1, 2)
        freq_output_proj = freq_output_proj + time_batched

        freq_output_mhsa = self.freq_mhsa(freq_output_proj)
        freq_output_mhsa = freq_output_mhsa + freq_output_proj

        freq_output = self.freq_ffn(
            self.freq_norm_2(freq_output_mhsa).transpose(1, 2)
        ).transpose(1, 2)
        freq_output = freq_output + freq_output_mhsa

        freq_batched = rearrange(
            freq_output, "(b t) f c -> (b f) t c", b=x.shape[0], t=x.shape[1]
        )
        time_output_proj = self.time_proj(
            self.time_norm(freq_batched).transpose(1, 2)
        ).transpose(1, 2)
        time_output_proj = 0.5 * time_output_proj + freq_batched

        time_output_mhsa = self.time_mhsa(time_output_proj)
        time_output_mhsa = time_output_mhsa + time_output_proj

        time_output = self.time_ffn(
            self.time_norm_2(time_output_mhsa).transpose(1, 2)
        ).transpose(1, 2)
        time_output = time_output + time_output_mhsa

        output = rearrange(
            time_output, "(b f) t c -> b t f c", b=x.shape[0], f=x.shape[2]
        )
        return output


class ModifiedSRCorrNetDecoderBlock(torch.nn.Module):
    def __init__(self, params: ModifiedSRCorrNetDecoderBlockParameters):
        super().__init__()

        self.freq_norm = torch.nn.LayerNorm(params.dim)
        self.freq_proj = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.dim,
                2 * params.dim_feedforward,
                params.kernel_size_feedforward,
                padding="same",
            ),
            SwiGLU(),
            torch.nn.Conv1d(
                params.dim_feedforward,
                params.dim,
                params.kernel_size_feedforward,
                padding="same",
            ),
        )
        self.freq_mhsa = torch.nn.Sequential(
            torch.nn.LayerNorm(params.dim), RoPESelfAttention(params.dim, params.nheads)
        )
        self.freq_norm_2 = torch.nn.LayerNorm(params.dim)
        self.freq_ffn = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.dim,
                2 * params.dim_feedforward,
                params.kernel_size_feedforward,
                padding="same",
            ),
            SwiGLU(),
            torch.nn.Conv1d(
                params.dim_feedforward,
                params.dim,
                params.kernel_size_feedforward,
                padding="same",
            ),
        )

        self.time_norm = torch.nn.LayerNorm(params.dim)
        self.time_proj = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.dim,
                2 * params.dim_feedforward,
                params.kernel_size_feedforward,
                padding="same",
            ),
            SwiGLU(),
            torch.nn.Conv1d(
                params.dim_feedforward,
                params.dim,
                params.kernel_size_feedforward,
                padding="same",
            ),
        )
        self.time_mhsa = torch.nn.Sequential(
            torch.nn.LayerNorm(params.dim), RoPESelfAttention(params.dim, params.nheads)
        )
        self.time_norm_2 = torch.nn.LayerNorm(params.dim)
        self.time_ffn = torch.nn.Sequential(
            torch.nn.Conv1d(
                params.dim,
                2 * params.dim_feedforward,
                params.kernel_size_feedforward,
                padding="same",
            ),
            SwiGLU(),
            torch.nn.Conv1d(
                params.dim_feedforward,
                params.dim,
                params.kernel_size_feedforward,
                padding="same",
            ),
        )

        self.speaker_mhsa = torch.nn.Sequential(
            torch.nn.LayerNorm(params.dim), SelfAttention(params.dim, params.nheads)
        )
        self.speaker_ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(params.dim),
            torch.nn.Linear(
                params.dim,
                4 * params.dim,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                4 * params.dim,
                params.dim,
            ),
        )

    def forward(self, x):
        # Assume x is B x K x T x F x C
        time_speaker_batched = rearrange(x, "b k t f c -> (b k t) f c")

        freq_output_proj = self.freq_proj(
            self.freq_norm(time_speaker_batched).transpose(1, 2)
        ).transpose(1, 2)
        freq_output_proj = freq_output_proj + time_speaker_batched

        freq_output_mhsa = self.freq_mhsa(freq_output_proj)
        freq_output_mhsa = freq_output_mhsa + freq_output_proj

        freq_output = self.freq_ffn(
            self.freq_norm_2(freq_output_mhsa).transpose(1, 2)
        ).transpose(1, 2)
        freq_output = freq_output + freq_output_mhsa

        freq_speaker_batched = rearrange(
            freq_output,
            "(b k t) f c -> (b k f) t c",
            b=x.shape[0],
            k=x.shape[1],
            t=x.shape[2],
        )
        time_output_proj = self.time_proj(
            self.time_norm(freq_speaker_batched).transpose(1, 2)
        ).transpose(1, 2)
        time_output_proj = time_output_proj + freq_speaker_batched

        time_output_mhsa = self.time_mhsa(time_output_proj)
        time_output_mhsa = time_output_mhsa + time_output_proj

        time_output = self.time_ffn(
            self.time_norm_2(time_output_mhsa).transpose(1, 2)
        ).transpose(1, 2)
        time_output = time_output + time_output_mhsa

        time_freq_batched = rearrange(
            time_output,
            "(b k f) t c -> (b t f) k c",
            b=x.shape[0],
            t=x.shape[2],
            f=x.shape[3],
        )
        chunk_size = 8192
        results = [
            self.speaker_mhsa(time_freq_batched[start : start + chunk_size])
            for start in range(0, len(time_freq_batched), chunk_size)
        ]
        speaker_output_mhsa = torch.cat(results, dim=0)
        speaker_output_mhsa = speaker_output_mhsa + time_freq_batched

        speaker_output = self.speaker_ffn(speaker_output_mhsa)
        speaker_output = speaker_output + speaker_output_mhsa

        output = rearrange(
            speaker_output,
            "(b t f) k c -> b k t f c",
            b=x.shape[0],
            t=x.shape[2],
            f=x.shape[3],
        )
        return output


class ModifiedSRCorrNet(torch.nn.Module):
    def __init__(self, params: ModifiedSRCorrNetParameters):
        super().__init__()

        self.num_speakers = params.num_speakers
        self.in_proj = torch.nn.Linear(2, params.dim)
        self.encoder = torch.nn.Sequential(
            *[
                ModifiedSRCorrNetEncoderBlock(params.encoder_block_parameters)
                for _ in range(params.num_encoder_blocks)
            ]
        )
        self.splitter = torch.nn.Sequential(
            torch.nn.Linear(params.dim, params.dim * params.num_speakers),
            torch.nn.ReLU(),
            torch.nn.Linear(
                params.dim * params.num_speakers, params.dim * params.num_speakers
            ),
        )
        self.splitter_layer_norm = torch.nn.LayerNorm(params.dim)
        self.decoder = torch.nn.Sequential(
            *[
                ModifiedSRCorrNetDecoderBlock(params.decoder_block_parameters)
                for _ in range(params.num_decoder_blocks)
            ]
        )
        self.out_proj = torch.nn.Linear(params.dim, 2)

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=params.n_fft,
            win_length=params.win_length,
            hop_length=params.hop_length,
            power=None,
        )
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=params.n_fft,
            win_length=params.win_length,
            hop_length=params.hop_length,
        )

    def forward(self, x):
        # assume x is waveform, B x T
        length = x.shape[1]
        x = torch.view_as_real(self.stft(x))
        x = rearrange(x, "b f t c -> b t f c")
        x = self.in_proj(x)

        encoder_output = self.encoder(x)
        splitter_output = self.splitter(encoder_output)
        splitter_output = rearrange(
            splitter_output, "b t f (k c) -> b k t f c", k=self.num_speakers
        )
        splitter_output = self.splitter_layer_norm(splitter_output)
        decoder_output = self.decoder(splitter_output)

        output = self.out_proj(decoder_output)
        output = rearrange(output, "b k t f c -> b k f t c")
        output = torch.view_as_complex(output.contiguous())
        output = self.istft(output, length=length)
        return output
