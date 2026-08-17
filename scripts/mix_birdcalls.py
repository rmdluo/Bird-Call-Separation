import argparse
import os
from pathlib import Path
import shutil
import torch
import torchaudio
import random
import json
import sys

sys.path.append("../src")
from processing.audio import (
    change_sample_rate,
    change_duration,
    convert_to_mono,
    mix_waveforms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brief description of what this script does."
    )

    parser.add_argument(
        "audio_dataset_path",
        type=Path,
        help="Path to the audio dataset.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("iNaturalistAudioMixed"),
        help="Path to the output. Default: iNaturalistAudioMixed",
    )

    parser.add_argument(
        "-s",
        "--sample-rate",
        type=int,
        default=8000,
        help="Sample rate to resample to. Default: 8000",
    )

    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=5,
        help="Number of seconds to use. Default: 5",
    )

    parser.add_argument(
        "-nb",
        "--num-birds",
        type=int,
        default=3,
        help="Number of birds to use per recording. Default: 3",
    )

    parser.add_argument(
        "-n",
        "--num-mixtures",
        type=int,
        default=50000,
        help="Number of mixtures to make. Default: 50000",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Value to seed python's random with. Default: 42",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do a dry run, where files will not be moved or deleted",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print(f"Arguments: {args}")

    if not args.output:
        print("Please specify an output path")
        exit(1)

    if args.verbose:
        print(f"Seeding with {args.seed}")
    random.seed(args.seed)

    audio_output_folder = args.output

    print(f"Input: {args.audio_dataset_path}")
    print(f"Output: {audio_output_folder}")

    if not args.dry_run and os.path.exists(audio_output_folder):
        if args.verbose:
            print(f"{audio_output_folder} exists, deleting...")
        if not args.dry_run:
            shutil.rmtree(audio_output_folder)

    if not args.dry_run:
        if args.verbose:
            print(f"Creating {audio_output_folder}")
        os.makedirs(audio_output_folder)

    all_species = os.listdir(args.audio_dataset_path)
    files_by_species = {
        species: sorted(list(args.audio_dataset_path.glob(f"**/{species}/*.wav")))
        for species in all_species
    }
    num_mixtures = 0
    while num_mixtures < args.num_mixtures:
        selected_species = random.choices(all_species, k=args.num_birds)

        audio_files = []
        for species in selected_species:
            audio_file = random.choice(files_by_species[species])
            audio_files.append(audio_file)

        if args.verbose:
            print(f"Mixing {audio_files}")

        if not args.dry_run:
            loaded_audio = [
                torchaudio.load(audio_file, normalize=True)
                for audio_file in audio_files
            ]

            skip_flag = False
            resampled_audios = []
            for i, (waveform, sample_rate) in enumerate(loaded_audio):
                if waveform.numel() == 0 or waveform.shape[-1] == 0:
                    if args.verbose:
                        print(
                            f"Empty audio file, removing from pool: {audio_files[i].resolve().as_uri()}"
                        )
                    files_by_species[selected_species[i]].remove(audio_files[i])
                    skip_flag = True
                    break

                resampled_waveform, sample_rate = change_sample_rate(
                    waveform, sample_rate, args.sample_rate
                )
                resampled_waveform, sample_rate = change_duration(
                    resampled_waveform, sample_rate, args.time
                )
                resampled_waveform = convert_to_mono(resampled_waveform)

                resampled_audios.append(resampled_waveform)

            if skip_flag:
                continue

            sources = torch.stack(resampled_audios, dim=0)
            mixed_waveform = mix_waveforms(sources)

            # Rescale so there is no clipping
            peak = mixed_waveform.abs().max().item()
            scale = 1
            if peak > 0.99:
                scale = 0.99 / peak
                sources = sources * scale
                mixed_waveform = mixed_waveform * scale

            torchaudio.save(
                audio_output_folder / f"mix_{num_mixtures}.wav",
                mixed_waveform,
                sample_rate,
            )

            metadata = {
                "mixture": str(audio_output_folder / f"mix_{num_mixtures}.wav"),
                "sources": [str(path) for path in audio_files],
                "scale": scale,
                "sample_rate": args.sample_rate,
                "duration": args.time,
            }
            with open(
                audio_output_folder / f"mix_{num_mixtures}.json",
                "w",
            ) as f:
                json.dump(metadata, f, indent=2)
        num_mixtures += 1


if __name__ == "__main__":
    main()
