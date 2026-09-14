#!/usr/bin/env python3
"""
PyTorch Lightning training script for the basic UNet model on the SatRain dataset.
All configuration is read from TOML files; the loss can be selected with --loss.
"""

import argparse
import logging

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from satrain_models import (
    SatRainDataModule,
    SatRainEstimationModule,
    create_unet,
)
from satrain_models.cli import (
    add_common_args,
    load_compute_config,
    load_dataset_config,
    resolve_loss_from_args,
    save_training_artifacts,
)

LOGGER = logging.getLogger("basic_unet_training")

MODEL_FAMILY = "unet"
BILINEAR = True


def main():
    """Training function"""

    parser = argparse.ArgumentParser(description="Train basic UNet model")
    # Positional compute config kept for backwards compatibility with
    # ``python train.py compute_adamw.toml``; ``--compute-config`` also works.
    parser.add_argument(
        "compute_config_positional",
        nargs="?",
        metavar="compute_config",
        help="Path to compute config file (default: compute.toml).",
        default=None,
    )
    add_common_args(parser)
    args = parser.parse_args()
    if args.compute_config_positional is not None:
        args.compute_config = args.compute_config_positional

    config = load_dataset_config(args)
    compute_config = load_compute_config(args)
    loss = resolve_loss_from_args(args, compute_config)

    # Create data module
    datamodule = SatRainDataModule(
        config=config,
        batch_size=compute_config.batch_size,
        num_workers=compute_config.num_workers,
        pin_memory=compute_config.pin_memory,
        persistent_workers=compute_config.persistent_workers,
    )

    # Create model; the loss dictates the number of output channels.
    unet_model = create_unet(
        n_channels=datamodule.num_features, n_outputs=loss.n_outputs, bilinear=BILINEAR
    )
    model_config = {"bilinear": BILINEAR, "n_outputs": loss.n_outputs}

    # Create Lightning module with dataset-aware naming
    experiment_prefix = config.get_experiment_name_prefix(MODEL_FAMILY)
    lightning_module = SatRainEstimationModule(
        model=unet_model,
        loss=loss,
        approach=compute_config.approach,
        learning_rate=compute_config.learning_rate,
        name=experiment_prefix,
    )
    experiment_name = lightning_module.experiment_name

    loggers = [TensorBoardLogger(save_dir="lightning_logs", name=experiment_name)]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=compute_config.max_epochs,
        accelerator=compute_config.accelerator,
        devices=compute_config.devices,
        strategy="auto",
        precision=compute_config.precision,
        logger=loggers,
        callbacks=lightning_module.default_callbacks(),
        log_every_n_steps=compute_config.log_every_n_steps,
        check_val_every_n_epoch=compute_config.check_val_every_n_epoch,
        accumulate_grad_batches=compute_config.accumulate_grad_batches,
        **compute_config.trainer_overrides(),
    )

    # Train the model
    LOGGER.info(f"Starting the training: {experiment_name}")
    trainer.fit(lightning_module, datamodule)

    # Save training metrics and model weights
    LOGGER.info("Training finished. Saving metrics.")
    save_training_artifacts(
        lightning_module,
        loggers[0].log_dir,
        config,
        compute_config,
        MODEL_FAMILY,
        model_config=model_config,
        model_name="basic_unet",
        study=args.study,
        num_parameters=unet_model.num_parameters,
        num_trainable_parameters=unet_model.num_trainable_parameters,
    )


if __name__ == "__main__":
    main()
