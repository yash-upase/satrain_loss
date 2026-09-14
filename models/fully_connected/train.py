#!/usr/bin/env python3
"""
PyTorch Lightning training script for the fully-connected (MLP) model on the
SatRain dataset (tabular format). The loss can be selected with --loss.
"""
import argparse
import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from satrain_models import (
    SatRainDataModule,
    SatRainEstimationModule,
    create_fully_connected,
    FullyConnectedConfig,
)
from satrain_models.cli import (
    add_common_args,
    load_compute_config,
    load_dataset_config,
    resolve_loss_from_args,
    save_training_artifacts,
)

LOGGER = logging.getLogger("basic_fully_connected_training")

MODEL_FAMILY = "fully_connected"


def main():
    """Training function"""
    parser = argparse.ArgumentParser(description="Train FullyConnected model")
    parser.add_argument("model_config", help="Path to model configuration TOML file")
    add_common_args(parser)
    args = parser.parse_args()

    # Load model configuration
    model_config_path = Path(args.model_config)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_config = FullyConnectedConfig.from_toml_file(model_config_path)
    LOGGER.info(f"Loaded model config from: {model_config_path}")

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

    # Create model using model config; the loss dictates the number of outputs.
    fully_connected_model = create_fully_connected(
        input_dim=datamodule.num_features,
        hidden_dims=model_config.hidden_dims,
        output_dim=loss.n_outputs,
        dropout=model_config.dropout
    )

    # Create experiment name that includes dataset configuration
    dataset_prefix = config.get_experiment_name_prefix(MODEL_FAMILY)
    full_experiment_name = f"{dataset_prefix}_{model_config.model_name}"

    # Create Lightning module with custom name
    lightning_module = SatRainEstimationModule(
        model=fully_connected_model,
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
        hidden_dims=str(model_config.hidden_dims),
        dropout=model_config.dropout,
        num_parameters=sum(p.numel() for p in fully_connected_model.parameters()),
        num_trainable_parameters=sum(p.numel() for p in fully_connected_model.parameters() if p.requires_grad),
    )


if __name__ == "__main__":
    main()
