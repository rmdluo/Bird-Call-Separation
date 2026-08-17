from pathlib import Path
import json
import torch
import torchaudio
from processing.audio import change_sample_rate, change_duration, convert_to_mono


class BirdCallMixture(torch.utils.data.Dataset):
    def __init__(self, dataset_root_path, split="train", with_species=False):
        self.dataset_root = Path(dataset_root_path)
        self.split = split
        self.with_species = with_species

        self.metadata_files = sorted(
            list((self.dataset_root / self.split).glob("**/*.json"))
        )

        self.species_encoder = {}
        self.species_decoder = {}

    def encode_species(self, species):
        return self.species_encoder[species]

    def decode_species(self, species_idx):
        return self.species_decoder[species_idx]

    def __len__(self):
        return len(self.metadata_files)

    def __getitem__(self, idx):
        metadata_uri = self.metadata_files[idx]

        with open(metadata_uri) as f:
            metadata = json.load(f)

        mixture_uri = Path(metadata["mixture"])

        source_uris = [Path(source_uri) for source_uri in metadata["sources"]]

        mixture_waveform, _ = torchaudio.load(mixture_uri, normalize=True)
        mixture_waveform = mixture_waveform.squeeze()

        source_waveforms = []
        species = []
        for source_uri in source_uris:
            source_waveform, source_sample_rate = torchaudio.load(
                source_uri, normalize=True
            )

            resampled_waveform, sample_rate = change_sample_rate(
                source_waveform, source_sample_rate, int(metadata["sample_rate"])
            )
            resampled_waveform, sample_rate = change_duration(
                resampled_waveform, sample_rate, int(metadata["duration"])
            )
            resampled_waveform = convert_to_mono(resampled_waveform)
            resampled_waveform = resampled_waveform * float(metadata["scale"])

            source_waveforms.append(resampled_waveform.squeeze())

            specimen = source_uri.parent.name
            if specimen not in self.species_encoder:
                self.species_encoder[specimen] = len(self.species_decoder)
                self.species_decoder[len(self.species_decoder)] = specimen
            species.append(self.species_encoder[specimen])

        if not self.with_species:
            return mixture_waveform, torch.stack(source_waveforms, dim=0)
        else:
            return mixture_waveform, torch.stack(source_waveforms, dim=0), torch.tensor(species)
