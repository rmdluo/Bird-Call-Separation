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

CHECKPOINT_PATH = "../models/modified_sr_corrnet_2/1786669770644132506/modified_sr_corrnet_final.tar"
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
train_parameters = checkpoint['train_parameters']
train_parameters.device = "cuda:0"
model_parameters = checkpoint['model_parameters']
model = ModifiedSRCorrNet(model_parameters)
model.load_state_dict(checkpoint['model_state_dict'])
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

sisnr_loss_fn = SISNRLoss(
    loss_clipping=None, eps=train_parameters.eps
)

sisnr = 0
same_species_sisnr = 0
samples = 0
same_species_samples = 0
for mixture, sources, species in tqdm(test_dl):
    has_duplicate_species = (species[:, 0] == species[:, 1]) | (species[:, 0] == species[:, 2]) | (species[:, 1] == species[:, 2])

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
print("Same Species Negative SISNR:", same_species_sisnr / same_species_samples)


# for run_name in runs:
#     run = config[run_name]
#     if not run["enable"]:
#         continue
#     train_params = ModifiedSRCorrNetTrainParameters(
#         lr=run["train_params"]["lr"],
#         batch_size=general["batch_size"],
#         epochs=general["epochs"],
#         num_workers=general["num_workers"],
#         loss_clipping=run["train_params"]["loss_clipping"],
#         eps=general["eps"],
#         model_save_path=general["model_save_path"] + str(time.time_ns()),
#         model_save_name="modified_sr_corrnet",
#         model_save_frequency=general["model_save_frequency"],
#         device=general["device"],
#         seed=general["seed"],
#         compile=general["compile"],
#     )

#     model_params = ModifiedSRCorrNetParameters(
#         n_fft=run["model_params"]["n_fft"],
#         win_length=run["model_params"]["win_length"],
#         hop_length=run["model_params"]["hop_length"],
#         num_encoder_blocks=run["model_params"]["num_encoder_blocks"],
#         num_decoder_blocks=run["model_params"]["num_decoder_blocks"],
#         num_speakers=run["model_params"]["num_speakers"],
#         dim=run["model_params"]["dim"],
#         encoder_block_parameters=ModifiedSRCorrNetEncoderBlockParameters(
#             dim=run["model_params"]["dim"],
#             nheads=run["model_params"]["nheads"],
#             dim_feedforward=run["model_params"]["dim_feedforward"],
#             kernel_size_feedforward=run["model_params"]["kernel_size_feedforward"],
#         ),
#         decoder_block_parameters=ModifiedSRCorrNetDecoderBlockParameters(
#             dim=run["model_params"]["dim"],
#             nheads=run["model_params"]["nheads"],
#             dim_feedforward=run["model_params"]["dim_feedforward"],
#             kernel_size_feedforward=run["model_params"]["kernel_size_feedforward"],
#         ),
#     )

#     train(train_dataset, val_dataset, test_dataset, model_params, train_params)

#     with open(
#         Path(train_params.model_save_path) / "config.yaml",
#         "w",
#         encoding="utf-8",
#     ) as file:
#         yaml.safe_dump(
#             {
#                 "general": general,
#                 "run-params": run,
#             },
#             file,
#             sort_keys=False,
#             default_flow_style=False,
#         )
