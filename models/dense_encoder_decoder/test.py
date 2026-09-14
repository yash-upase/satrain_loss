#!/usr/bin/env python3
"""
Evaluation script for the DenseEncoderDecoder (UNet++) model on the SatRain
test domains. The dataset, model and loss configuration are read from the
saved model file.
"""
import argparse
import logging
from pathlib import Path

from satrain_models import SatRainDataModule, SatRainEstimationModule, get_loss
from satrain_models.cli import (
    add_common_args,
    load_compute_config,
    load_satrain_config_for_test,
    load_saved_model,
    result_attrs,
    run_evaluation,
    save_test_results,
)
from satrain_models.encoder_decoder import (
    EncoderDecoderConfig,
    Conv2dBnReLU,
    Conv2dLnGELU,
    Conv2dLnReLU,
    ResNeXtBlock,
    InvertedBottleneck,
)
from satrain_models.unetpp import DenseEncoderDecoder

LOGGER = logging.getLogger(__name__)

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
    """Main testing function - all configuration from TOML files and saved model."""
    parser = argparse.ArgumentParser(description="Test DenseEncoderDecoder model")
    parser.add_argument("model", help="Path to model weight or checkpoint file.")
    add_common_args(parser, testing=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    saved = load_saved_model(model_path)
    satrain_config = load_satrain_config_for_test(saved)
    loss = get_loss(**saved["loss_config"])
    LOGGER.info(f"Loaded model file {model_path} (loss: {loss.name})")

    if saved["model_config"] is not None:
        model_config = EncoderDecoderConfig(**saved["model_config"])
    else:
        model_config_path = Path("model.toml")
        if not model_config_path.exists():
            raise FileNotFoundError(
                "No model configuration found. Either save model with model_config "
                "or provide model.toml file."
            )
        model_config = EncoderDecoderConfig.from_toml_file(model_config_path)
    model_config.out_channels = loss.n_outputs

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
    encoder_decoder_model = create_encoder_decoder_from_config(
        model_config=model_config,
        num_features=satrain_config.num_features,
    )
    encoder_decoder_model.load_state_dict(saved["state_dict"])
    LOGGER.info(f"Loaded {model_config.model_name} with {encoder_decoder_model.num_parameters:,} parameters")

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=encoder_decoder_model,
        loss=loss,
        approach=compute_config.approach,
    )
    experiment_name = model_path.stem

    results = run_evaluation(
        lightning_module,
        data_module,
        satrain_config,
        compute_config,
        batch_size=4 * compute_config.batch_size,
    )

    attrs = result_attrs(
        satrain_config,
        compute_config,
        loss,
        experiment_name,
        MODEL_FAMILY,
        model_name=model_config.model_name,
        study=args.study,
        block_factory=model_config.block_factory,
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        channels=str(model_config.channels),
        depths=str(model_config.depths),
        num_parameters=encoder_decoder_model.num_parameters,
        num_trainable_parameters=encoder_decoder_model.num_trainable_parameters,
    )
    save_test_results(results, experiment_name, attrs)


if __name__ == "__main__":
    main()
