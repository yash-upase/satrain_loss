#!/usr/bin/env python3
"""
Script for training a generic dense encoder-decoder (UNet++) model on the
SatRain dataset. The loss can be selected with --loss.
"""
import argparse
import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from satrain_models import SatRainDataModule, SatRainEstimationModule
from satrain_models.cli import (
    add_common_args,
    load_compute_config,
    load_dataset_config,
    resolve_loss_from_args,
    save_training_artifacts,
)
from satrain_models.encoder_decoder import (
    EncoderDecoderConfig,
    Conv2dBnReLU,
    Conv2dLnGELU,
    Conv2dLnReLU,
    ResNeXtBlock,
    InvertedBottleneck
)
from satrain_models.unetpp import DenseEncoderDecoder


LOGGER = logging.getLogger("dense_encoder_decoder_training")

MODEL_FAMILY = "dense_encoder_decoder"

# Available block classes mapping
BLOCK_FACTORIES = {
    "Conv2dBnReLU": Conv2dBnReLU,
    "Conv2dLnGELU": Conv2dLnGELU,
    "Conv2dLnReLU": Conv2dLnReLU,
    "ResNeXtBlock": ResNeXtBlock,
    "InvertedBottleneck": InvertedBottleneck,
}


def create_encoder_decoder_from_config(
        model_config: EncoderDecoderConfig,
        num_features: int
) -> DenseEncoderDecoder:
    """Create DenseEncoderDecoder model from configuration."""
    block_factory = BLOCK_FACTORIES[model_config.block_factory]

    return DenseEncoderDecoder(
        block_factory=block_factory,
        in_channels=num_features,
        channels=model_config.channels,
        depths=model_config.depths,
        out_channels=model_config.out_channels,
    )


def main():
    """Main training function - all configuration from TOML files."""
    parser = argparse.ArgumentParser(description="Train DenseEncoderDecoder model")
    parser.add_argument("model_config", help="Path to model configuration TOML file")
    add_common_args(parser)
    args = parser.parse_args()

    # Load model configuration
    model_config_path = Path(args.model_config)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_config = EncoderDecoderConfig.from_toml_file(model_config_path)
    LOGGER.info(f"Loaded model config from: {model_config_path}")

    config = load_dataset_config(args)
    compute_config = load_compute_config(args)
    loss = resolve_loss_from_args(args, compute_config)

    model_config.in_channels = config.num_features
    model_config.out_channels = loss.n_outputs

    # Create data module
    datamodule = SatRainDataModule(
        config=config,
        batch_size=compute_config.batch_size,
        num_workers=compute_config.num_workers,
        pin_memory=compute_config.pin_memory,
        persistent_workers=compute_config.persistent_workers,
    )

    # Create model
    encoder_decoder_model = create_encoder_decoder_from_config(
        model_config=model_config,
        num_features=datamodule.num_features,
    )

    LOGGER.info(f"Created {model_config.model_name} with {encoder_decoder_model.num_parameters:,} parameters")

    # Create experiment name that includes dataset configuration
    dataset_prefix = config.get_experiment_name_prefix(MODEL_FAMILY)
    full_experiment_name = f"{dataset_prefix}_{model_config.model_name}"

    # Create Lightning module with custom name
    lightning_module = SatRainEstimationModule(
        model=encoder_decoder_model,
        loss=loss,
        approach=compute_config.approach,
        learning_rate=compute_config.learning_rate,
        name=full_experiment_name,
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
    LOGGER.info(f"Starting training: {experiment_name}")
    trainer.fit(lightning_module, datamodule)

    # Save training metrics and model weights
    LOGGER.info("Training finished. Saving metrics.")
    save_training_artifacts(
        lightning_module,
        loggers[0].log_dir,
        config,
        compute_config,
        MODEL_FAMILY,
        model_config=model_config.to_dict(),
        model_name=model_config.model_name,
        study=args.study,
        block_factory=model_config.block_factory,
        num_parameters=encoder_decoder_model.num_parameters,
        num_trainable_parameters=encoder_decoder_model.num_trainable_parameters,
    )


if __name__ == "__main__":
    main()
