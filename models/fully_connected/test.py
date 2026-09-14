#!/usr/bin/env python3
"""
Evaluation script for the fully-connected (MLP) model on the SatRain test
domains (tabular input). The dataset, model and loss configuration are read
from the saved model file.
"""
import argparse
import logging
from pathlib import Path

from satrain_models import (
    SatRainEstimationModule, create_fully_connected, SatRainDataModule,
    FullyConnectedConfig, get_loss,
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

MODEL_FAMILY = "fully_connected"


def main():
    """Evaluate a trained model on all test domains."""
    parser = argparse.ArgumentParser(description="Test FullyConnected model")
    parser.add_argument("model", help="Path to model weight or checkpoint file.")
    add_common_args(parser, testing=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    saved = load_saved_model(model_path)
    satrain_config = load_satrain_config_for_test(saved)
    loss = get_loss(**saved["loss_config"])
    LOGGER.info(f"Loaded model file {model_path} (loss: {loss.name})")

    if saved["model_config"] is not None:
        model_config = FullyConnectedConfig(**saved["model_config"])
    else:
        model_config = FullyConnectedConfig()
        LOGGER.info("Using default FullyConnected config (model file has no 'model_config')")

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
    fully_connected_model = create_fully_connected(
        input_dim=satrain_config.num_features,
        hidden_dims=model_config.hidden_dims,
        output_dim=loss.n_outputs,
        dropout=model_config.dropout
    )
    fully_connected_model.load_state_dict(saved["state_dict"])

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=fully_connected_model,
        loss=loss,
        approach=compute_config.approach,
    )
    experiment_name = model_path.stem

    results = run_evaluation(
        lightning_module,
        data_module,
        satrain_config,
        compute_config,
        input_data_format="tabular",
    )

    attrs = result_attrs(
        satrain_config,
        compute_config,
        loss,
        experiment_name,
        MODEL_FAMILY,
        model_name=model_config.model_name,
        study=args.study,
        hidden_dims=str(model_config.hidden_dims),
        dropout=model_config.dropout,
        num_parameters=sum(p.numel() for p in fully_connected_model.parameters()),
        num_trainable_parameters=sum(p.numel() for p in fully_connected_model.parameters() if p.requires_grad),
    )
    save_test_results(results, experiment_name, attrs)


if __name__ == '__main__':
    main()
