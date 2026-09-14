#!/usr/bin/env python3
"""
Evaluation script for the basic UNet model on the SatRain test domains.
The dataset and loss configuration are read from the saved model file.
"""

import argparse
import logging
from pathlib import Path

from satrain_models import (
    SatRainDataModule,
    SatRainEstimationModule,
    create_unet,
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

LOGGER = logging.getLogger(__name__)

MODEL_FAMILY = "unet"


def main():
    """Evaluate a trained model on all test domains."""
    parser = argparse.ArgumentParser(description="Test basic UNet model")
    parser.add_argument("model", help="Path to model weight or checkpoint file.")
    add_common_args(parser, testing=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    saved = load_saved_model(model_path)
    satrain_config = load_satrain_config_for_test(saved)
    loss = get_loss(**saved["loss_config"])
    model_config = saved["model_config"] or {"bilinear": True, "n_outputs": loss.n_outputs}
    LOGGER.info(f"Loaded model file {model_path} (loss: {loss.name})")

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
    unet_model = create_unet(
        n_channels=satrain_config.num_features,
        n_outputs=loss.n_outputs,
        bilinear=model_config.get("bilinear", True),
    )
    unet_model.load_state_dict(saved["state_dict"])

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=unet_model,
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
        model_name="basic_unet",
        study=args.study,
        num_parameters=unet_model.num_parameters,
        num_trainable_parameters=unet_model.num_trainable_parameters,
    )
    save_test_results(results, experiment_name, attrs)


if __name__ == "__main__":
    main()
