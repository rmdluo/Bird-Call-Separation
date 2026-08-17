from dataclasses import dataclass
from tqdm import tqdm
import os
import torch

from datasets.mixture import BirdCallMixture
from models.modified_sr_corrnet import (
    ModifiedSRCorrNet,
    ModifiedSRCorrNetParameters,
)
from train.losses import SISNRLoss


@dataclass
class ModifiedSRCorrNetTrainParameters:
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    num_workers: int = 16
    loss_clipping: float = 30.0
    eps: float = 1e-5
    model_save_path: str = "photo_model/"
    model_save_name: str = "photo_model"
    model_save_frequency: int = 5
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    seed: int = 42
    compile: bool = True


def train(
    train_dataset: BirdCallMixture,
    val_dataset: BirdCallMixture,
    test_dataset: BirdCallMixture,
    params: ModifiedSRCorrNetParameters,
    train_parameters: ModifiedSRCorrNetTrainParameters,
):
    train_generator = torch.Generator()
    train_generator.manual_seed(train_parameters.seed)
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_parameters.batch_size,
        shuffle=True,
        num_workers=train_parameters.num_workers,
        pin_memory=True,
        generator=train_generator,
    )

    val_generator = torch.Generator()
    val_generator.manual_seed(train_parameters.seed)
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_parameters.batch_size,
        num_workers=train_parameters.num_workers,
        pin_memory=True,
        generator=val_generator,
    )

    if test_dataset is not None:
        test_generator = torch.Generator()
        test_generator.manual_seed(train_parameters.seed)
        test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=train_parameters.batch_size,
            num_workers=train_parameters.num_workers,
            pin_memory=True,
            generator=test_generator,
        )

    model = ModifiedSRCorrNet(params).to(train_parameters.device)
    if train_parameters.compile:
        model.compile()

    optimizer = torch.optim.AdamW(model.parameters(), train_parameters.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_parameters.epochs, eta_min=train_parameters.lr / 100
    )
    sisnr_loss_fn = SISNRLoss(
        loss_clipping=train_parameters.loss_clipping, eps=train_parameters.eps
    )

    train_losses = []
    val_losses = []
    for epoch in range(train_parameters.epochs):
        # Train
        model.train()
        running_total_loss = 0
        batches = 0
        for mixture, sources in tqdm(train_dl):
            mixture = mixture.to(train_parameters.device)
            sources = sources.to(train_parameters.device)
            optimizer.zero_grad()
            model_output = model(mixture)
            loss = 0
            for i in range(len(mixture)):
                sample_loss, _ = sisnr_loss_fn(model_output[i], sources[i])
                loss += sample_loss
            loss /= len(mixture)
            loss.backward()
            optimizer.step()
            running_total_loss += loss.detach().cpu().item()
            batches += 1

        train_losses.append(running_total_loss / batches)

        # Val
        with torch.no_grad():
            model.eval()
            running_total_loss = 0
            batches = 0
            for mixture, sources in tqdm(val_dl):
                mixture = mixture.to(train_parameters.device)
                sources = sources.to(train_parameters.device)
                model_output = model(mixture)
                loss = 0
                for i in range(len(mixture)):
                    sample_loss, _ = sisnr_loss_fn(model_output[i], sources[i])
                    loss += sample_loss
                loss /= len(mixture)
                running_total_loss += loss.detach().cpu().item()
                batches += 1

            val_losses.append(running_total_loss / batches)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{train_parameters.epochs} train results: total loss {train_losses[-1]}, lr {scheduler.get_last_lr()[0]}"
        )
        print(
            f"Epoch {epoch+1}/{train_parameters.epochs} val results: total loss {val_losses[-1]}"
        )

        if (epoch + 1) % train_parameters.model_save_frequency == 0:
            if not os.path.exists(train_parameters.model_save_path):
                os.makedirs(train_parameters.model_save_path)

            torch.save(
                {
                    "train_parameters": train_parameters,
                    "model_parameters": params,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lrscheduler_state_dict": scheduler.state_dict(),
                    "train_losses": {
                        "total_loss": train_losses,
                    },
                    "val_losses": {
                        "total_loss": val_losses,
                    },
                    "train_generator": train_generator.get_state(),
                },
                os.path.join(
                    train_parameters.model_save_path,
                    f"{train_parameters.model_save_name}_epoch_{epoch}.tar",
                ),
            )

    if test_dataset is not None:
        # Test
        with torch.no_grad():
            model.eval()
            running_total_loss = 0
            batches = 0
            for mixture, sources in tqdm(test_dl):
                mixture = mixture.to(train_parameters.device)
                sources = sources.to(train_parameters.device)
                model_output = model(mixture)
                loss = 0
                for i in range(len(mixture)):
                    sample_loss, _ = sisnr_loss_fn(model_output[i], sources[i])
                    loss += sample_loss
                loss /= len(mixture)
                running_total_loss += loss.detach().cpu().item()
                batches += 1

            test_loss = running_total_loss / batches

        print(
            f"Epoch {epoch+1}/{train_parameters.epochs} test results: total loss {test_loss}"
        )

    if not os.path.exists(train_parameters.model_save_path):
        os.makedirs(train_parameters.model_save_path)

    torch.save(
        {
            "train_parameters": train_parameters,
            "model_parameters": params,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lrscheduler_state_dict": scheduler.state_dict(),
            "train_losses": {
                "total_loss": train_losses,
            },
            "val_losses": {
                "total_loss": val_losses,
            },
            "train_generator": train_generator.get_state(),
        },
        os.path.join(
            train_parameters.model_save_path,
            f"{train_parameters.model_save_name}_final.tar",
        ),
    )
