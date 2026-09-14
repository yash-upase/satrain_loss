"""
Tests for the satrain_models.lightning module.
"""

from pathlib import Path

import lightning as L
import numpy as np
import pytest
import satrain
import satrain.data
import torch
from torch import nn

from satrain_models import create_unet
from satrain_models.config import ComputeConfig, SatRainConfig
from satrain_models.datamodule import SatRainDataModule
from satrain_models.lightning import SatRainEstimationModule


@pytest.fixture
def cpu_compute_config():
    config = ComputeConfig(
        approach="adamw_reduce_on_plateau",
        learning_rate=5e-4,
        batch_size=1,
        num_workers=0,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        precision="32",
    )
    return config


@pytest.fixture
def satrain_config_spatial():
    config = SatRainConfig.from_dict(
        {
            "base_sensor": "gmi",
            "retrieval_input": [{"name": "gmi", "include_angles": False, "nan": -1.5}],
            "geometry": "on_swath",
            "subset": "xs",
            "format": "spatial",
        }
    )
    return config


def test_lightning_module_spatial(
    satrain_config_spatial, cpu_compute_config, satrain_test_data
):
    """
    Test lightning module and evaluator in spatial format using a UNet.
    """
    data_module = SatRainDataModule(
        satrain_config_spatial,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    data_module.setup()

    # Test propagation
    model = SatRainEstimationModule(
        create_unet(13, 1, True),
        approach=cpu_compute_config.approach,
    )
    dl = data_module.train_dataloader()

    # Test propagation through network.
    x, y = next(iter(dl))
    with torch.no_grad():
        pred = model(x)
    assert pred.shape == (1, 1, 64, 64)

    # Test training
    trainer = L.Trainer(
        max_epochs=cpu_compute_config.max_epochs,
        accelerator=cpu_compute_config.accelerator,
        devices=cpu_compute_config.devices,
        strategy="auto",
        precision=cpu_compute_config.precision,
    )
    trainer.fit(model, data_module)

    # Test evaluation
    evaluator = data_module.get_evaluator("korea")
    retrieval_fn = model.get_retrieval_fn(satrain_config_spatial, cpu_compute_config)
    results = evaluator.evaluate_scene(
        0,
        tile_size=(64, 64),
        batch_size=1,
        overlap=8,
        retrieval_fn=retrieval_fn,
        input_data_format="spatial",
        track=False,
    )
    assert np.any(np.isfinite(results.surface_precip.data))


@pytest.fixture
def satrain_config_tabular():
    config = SatRainConfig.from_dict(
        {
            "base_sensor": "gmi",
            "retrieval_input": [{"name": "gmi", "include_angles": False, "nan": -1.5}],
            "geometry": "on_swath",
            "subset": "xs",
            "format": "tabular",
        }
    )
    return config


def test_lightning_module_tabular(
    satrain_config_tabular, cpu_compute_config, monkeypatch
):
    """
    Test lightning module and evaluator in tabular format using a simple MLP.
    """
    monkeypatch.setattr(satrain.config, "CONFIG_DIR", Path("."))

    data_module = SatRainDataModule(
        satrain_config_tabular,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    data_module.setup()

    # Test propagation
    model = SatRainEstimationModule(
        nn.Linear(13, 1),
        approach=cpu_compute_config.approach,
    )
    dl = data_module.train_dataloader()

    # Test propagation through network.
    x, y = next(iter(dl))
    with torch.no_grad():
        pred = model(x)

    assert pred.shape == (32, 1)

    # Test training
    trainer = L.Trainer(
        max_epochs=cpu_compute_config.max_epochs,
        accelerator=cpu_compute_config.accelerator,
        devices=cpu_compute_config.devices,
        strategy="auto",
        precision=cpu_compute_config.precision,
    )
    trainer.fit(model, data_module)

    # Test evaluation
    evaluator = data_module.get_evaluator("korea")
    retrieval_fn = model.get_retrieval_fn(satrain_config_tabular, cpu_compute_config)
    results = evaluator.evaluate_scene(
        0,
        tile_size=(64, 64),
        batch_size=256,
        overlap=8,
        retrieval_fn=retrieval_fn,
        input_data_format="tabular",
        track=False,
    )
    assert np.any(np.isfinite(results.surface_precip.data))


@pytest.mark.parametrize("loss_name", ["quantile", "tweedie", "hybrid"])
def test_lightning_module_multi_output_losses(
    satrain_config_spatial, cpu_compute_config, satrain_test_data, loss_name
):
    """
    Losses that change the number of model outputs or apply a link function
    train and evaluate through the same path as MSE.
    """
    from satrain_models import get_loss

    loss = get_loss(loss_name)
    data_module = SatRainDataModule(
        satrain_config_spatial,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    data_module.setup()

    model = SatRainEstimationModule(
        create_unet(13, loss.n_outputs, True),
        loss=loss,
        approach=cpu_compute_config.approach,
    )
    assert loss_name in model.experiment_name

    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, data_module)
    assert np.isfinite(trainer.callback_metrics["val/loss"].item())
    assert "val/mse" in trainer.callback_metrics

    evaluator = data_module.get_evaluator("korea")
    retrieval_fn = model.get_retrieval_fn(satrain_config_spatial, cpu_compute_config)
    results = evaluator.evaluate_scene(
        0,
        tile_size=(64, 64),
        batch_size=1,
        overlap=8,
        retrieval_fn=retrieval_fn,
        input_data_format="spatial",
        track=False,
    )
    assert np.any(np.isfinite(results.surface_precip.data))
    if loss.task == "hybrid":
        assert "probability_of_precip" in results
