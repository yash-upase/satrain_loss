#!/usr/bin/env python3
"""
Evaluation script for the EfficientNetV2 model on the SatRain test domains.
The dataset, model and loss configuration are read from the saved model file.
"""
import argparse
import logging
from pathlib import Path

from satrain_models import (
    SatRainEstimationModule, create_efficient_net_v2, SatRainDataModule,
    EfficientNetV2Config, get_loss,
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

MODEL_FAMILY = "efficient_net_v2"


def main():
    """Main evaluation function - all configuration from the saved model."""
    parser = argparse.ArgumentParser(description="Test EfficientNetV2 model")
    parser.add_argument("model", help="Path to model weight or checkpoint file (.pt, .pth, .ckpt)")
    add_common_args(parser, testing=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    saved = load_saved_model(model_path)
    satrain_config = load_satrain_config_for_test(saved)
    loss = get_loss(**saved["loss_config"])
    LOGGER.info(f"Loaded model file {model_path} (loss: {loss.name})")

    if saved["model_config"] is not None:
        model_config = EfficientNetV2Config(**saved["model_config"])
    else:
        model_config = EfficientNetV2Config()
        LOGGER.info("Using default EfficientNetV2 config (model file has no 'model_config')")

    compute_config = load_compute_config(args)

    # Create data module
    data_module = SatRainDataModule(
        config=satrain_config,
        batch_size=compute_config.batch_size,
        num_workers=compute_config.num_workers,
        pin_memory=compute_config.pin_memory,
        persistent_workers=compute_config.persistent_workers,
    )

    # Create EfficientNetV2 model (no pretrained download needed for evaluation)
    efficient_net_model = create_efficient_net_v2(
        n_channels=satrain_config.num_features,
        n_outputs=loss.n_outputs,
        model_size=model_config.model_size,
        pretrained=False,
    )
    efficient_net_model.load_state_dict(saved["state_dict"])

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=efficient_net_model,
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
        model_size=model_config.model_size,
        pretrained=str(model_config.pretrained),
        num_parameters=efficient_net_model.num_parameters,
        num_trainable_parameters=efficient_net_model.num_trainable_parameters,
    )
    save_test_results(results, experiment_name, attrs)


if __name__ == '__main__':
    main()
