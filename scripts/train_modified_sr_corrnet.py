import yaml
import sys

sys.path.append("../src")

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "modified_sr_corrnet_params.yaml"

with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

import time
from pathlib import Path
import torch

from datasets.mixture import BirdCallMixture
from train.train_modified_sr_corrnet import train, ModifiedSRCorrNetTrainParameters
from models.modified_sr_corrnet import (
    ModifiedSRCorrNetParameters,
    ModifiedSRCorrNetEncoderBlockParameters,
    ModifiedSRCorrNetDecoderBlockParameters,
)

general = config["general"]
runs = [key for key in config.keys() if key.startswith("run-")]

DATASET_PATH = "../data/iNaturalistAudioSubset"
train_dataset = BirdCallMixture(DATASET_PATH, split="train")
val_dataset = BirdCallMixture(DATASET_PATH, split="val")
test_dataset = BirdCallMixture(DATASET_PATH, split="test")

for run_name in runs:
    run = config[run_name]
    if not run["enable"]:
        continue
    train_params = ModifiedSRCorrNetTrainParameters(
        lr=run["train_params"]["lr"],
        batch_size=general["batch_size"],
        epochs=general["epochs"],
        num_workers=general["num_workers"],
        loss_clipping=run["train_params"]["loss_clipping"],
        eps=general["eps"],
        model_save_path=general["model_save_path"] + str(time.time_ns()),
        model_save_name="modified_sr_corrnet",
        model_save_frequency=general["model_save_frequency"],
        device=general["device"],
        seed=general["seed"],
        compile=general["compile"],
    )

    model_params = ModifiedSRCorrNetParameters(
        n_fft=run["model_params"]["n_fft"],
        win_length=run["model_params"]["win_length"],
        hop_length=run["model_params"]["hop_length"],
        num_encoder_blocks=run["model_params"]["num_encoder_blocks"],
        num_decoder_blocks=run["model_params"]["num_decoder_blocks"],
        num_speakers=run["model_params"]["num_speakers"],
        dim=run["model_params"]["dim"],
        encoder_block_parameters=ModifiedSRCorrNetEncoderBlockParameters(
            dim=run["model_params"]["dim"],
            nheads=run["model_params"]["nheads"],
            dim_feedforward=run["model_params"]["dim_feedforward"],
            kernel_size_feedforward=run["model_params"]["kernel_size_feedforward"],
        ),
        decoder_block_parameters=ModifiedSRCorrNetDecoderBlockParameters(
            dim=run["model_params"]["dim"],
            nheads=run["model_params"]["nheads"],
            dim_feedforward=run["model_params"]["dim_feedforward"],
            kernel_size_feedforward=run["model_params"]["kernel_size_feedforward"],
        ),
    )

    train(train_dataset, val_dataset, test_dataset, model_params, train_params)

    with open(
        Path(train_params.model_save_path) / "config.yaml",
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            {
                "general": general,
                "run-params": run,
            },
            file,
            sort_keys=False,
            default_flow_style=False,
        )
