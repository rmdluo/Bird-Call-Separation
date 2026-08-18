import yaml
import sys

sys.path.append("../src")

import time
from pathlib import Path
import torch
from tqdm import tqdm

from datasets.mixture import BirdCallMixture
from train.train_modified_sr_corrnet import ModifiedSRCorrNetTrainParameters
from train.losses import SISNRLoss
from models.modified_sr_corrnet import (
    ModifiedSRCorrNet,
    ModifiedSRCorrNetParameters,
    ModifiedSRCorrNetEncoderBlockParameters,
    ModifiedSRCorrNetDecoderBlockParameters,
)

CHECKPOINT_PATH = (
    "../models/modified_sr_corrnet_2/1786669770644132506/modified_sr_corrnet_final.tar"
)
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
train_parameters = checkpoint["train_parameters"]
train_parameters.device = "cuda:0"
model_parameters = checkpoint["model_parameters"]
model = ModifiedSRCorrNet(model_parameters)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(train_parameters.device)

DATASET_PATH = "../data/iNaturalistAudioSubset"
test_dataset = BirdCallMixture(DATASET_PATH, split="test", with_species=True)
test_generator = torch.Generator()
test_generator.manual_seed(train_parameters.seed)
test_dl = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=train_parameters.batch_size,
    num_workers=train_parameters.num_workers,
    pin_memory=True,
    generator=test_generator,
)

sisnr_loss_fn = SISNRLoss(loss_clipping=None, eps=train_parameters.eps)

sisnr = 0
same_species_sisnr = 0
samples = 0
same_species_samples = 0
for mixture, sources, species in tqdm(test_dl):
    has_duplicate_species = (
        (species[:, 0] == species[:, 1])
        | (species[:, 0] == species[:, 2])
        | (species[:, 1] == species[:, 2])
    )

    mixture = mixture.to(train_parameters.device)
    sources = sources.to(train_parameters.device)
    with torch.no_grad():
        model_output = model(mixture)

    for i in range(len(mixture)):
        sample_loss, _ = sisnr_loss_fn(model_output[i], sources[i])
        sisnr += sample_loss.cpu().item()
        samples += 1

        if has_duplicate_species[i]:
            same_species_sisnr += sample_loss.cpu().item()
            same_species_samples += 1

print("Total Negative SISNR:", sisnr / samples)
print("Number of samples:", samples)
print("Same Species Negative SISNR:", same_species_sisnr / same_species_samples)
print("Number of same-species samples:", same_species_samples)
