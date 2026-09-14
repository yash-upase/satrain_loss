#!/usr/bin/env python3
"""
Evaluation script for the ResNet model on the SatRain test domains.
The dataset, model and loss configuration are read from the saved model file.

Created: January 2026
questions: veljko.petkovic@umd.edu
"""

import argparse
import logging
from pathlib import Path

from satrain_models import (
    SatRainConfig,
    SatRainDataModule,
    SatRainEstimationModule,
    ResNetConfig,
    create_resnet_from_config,
    get_loss,
)
from satrain_models.cli import (
    add_common_args,
    load_compute_config,
    load_satrain_config_for_test,
    load_saved_model,
    result_attrs,
    run_evaluation,
    save_test_results,
)

# (veljko) Override tile_size for ResNet model
# Use 64x64 tiles which fit within GMI swath width (221 pixels)
RESNET_TILE_SIZE = (64, 64)
SatRainConfig.tile_size = property(lambda self: RESNET_TILE_SIZE)

LOGGER = logging.getLogger(__name__)

MODEL_FAMILY = "resnet"


def detect_architecture(state: dict, n_channels: int, n_outputs: int) -> ResNetConfig:
    """Infer the ResNet layout from a state dict (legacy files without model_config)."""
    has_layer4 = any('layer4' in key for key in state.keys())
    n_layers = 4 if has_layer4 else 3
    layer1_blocks = len([k for k in state.keys() if k.startswith('layer1.') and '.conv1.' in k])
    blocks_per_layer = max(layer1_blocks, 2)
    LOGGER.info(f"Detected model architecture: n_layers={n_layers}, blocks_per_layer={blocks_per_layer}")
    return ResNetConfig(
        n_channels=n_channels,
        n_outputs=n_outputs,
        blocks_per_layer=blocks_per_layer,
        n_layers=n_layers,
    )


def main():
    """Main testing function - evaluates model on all test domains."""
    parser = argparse.ArgumentParser(description="Test ResNet model")
    parser.add_argument("model", help="Path to model weight or checkpoint file.")
    add_common_args(parser, testing=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    saved = load_saved_model(model_path)
    satrain_config = load_satrain_config_for_test(saved)
    loss = get_loss(**saved["loss_config"])
    LOGGER.info(f"Loaded model file {model_path} (loss: {loss.name})")

    if saved["model_config"] is not None:
        model_config = ResNetConfig(**saved["model_config"])
    else:
        model_config = detect_architecture(
            saved["state_dict"], satrain_config.num_features, loss.n_outputs
        )
    model_config.n_outputs = loss.n_outputs

    compute_config = load_compute_config(args)

    # Create data module
    data_module = SatRainDataModule(
        config=satrain_config,
        batch_size=compute_config.batch_size,
        num_workers=compute_config.num_workers,
        pin_memory=compute_config.pin_memory,
        persistent_workers=compute_config.persistent_workers,
    )

    # Create model
    resnet_model = create_resnet_from_config(
        model_config=model_config,
        num_features=satrain_config.num_features,
    )
    resnet_model.load_state_dict(saved["state_dict"])

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=resnet_model,
        loss=loss,
        approach=compute_config.approach,
    )
    experiment_name = model_path.stem

    results = run_evaluation(lightning_module, data_module, satrain_config, compute_config)

    attrs = result_attrs(
        satrain_config,
        compute_config,
        loss,
        experiment_name,
        MODEL_FAMILY,
        model_name=model_config.model_name,
        study=args.study,
        blocks_per_layer=model_config.blocks_per_layer,
        n_layers=model_config.n_layers,
        base_channels=model_config.base_channels,
        num_parameters=sum(p.numel() for p in resnet_model.parameters()),
        num_trainable_parameters=sum(p.numel() for p in resnet_model.parameters() if p.requires_grad),
    )
    save_test_results(results, experiment_name, attrs)


if __name__ == "__main__":
    main()
