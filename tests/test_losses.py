"""
Tests for satrain_models.losses and the loss handling in the Lightning module.
"""

import math

import pytest
import torch
from torch import nn

from satrain_models.losses import (
    LOSSES,
    Loss,
    TorchLossAdapter,
    get_loss,
    load_loss_toml,
    register_loss,
    resolve_loss,
)
from satrain_models.lightning import SatRainEstimationModule
from satrain_models.config import ComputeConfig

PHASE1 = ["mse", "mae", "huber", "logcosh", "weighted_mse", "tweedie", "quantile"]


@pytest.fixture(params=sorted(LOSSES))
def loss_name(request):
    return request.param


def _module(loss):
    return SatRainEstimationModule(nn.Identity(), loss=loss, approach="adamw")


def test_phase1_losses_registered():
    for name in PHASE1:
        assert name in LOSSES


@pytest.mark.parametrize("shape", ["spatial", "tabular"])
def test_loss_masking_and_gradients(loss_name, shape):
    """Every loss: finite value, gradient flows, NaN targets are ignored."""
    loss = get_loss(loss_name)
    n_out = loss.n_outputs
    if shape == "spatial":
        pred = torch.randn(4, n_out, 16, 16, requires_grad=True)
        y = torch.rand(4, 16, 16) * 5
    else:
        pred = torch.randn(32, n_out, requires_grad=True)
        y = torch.rand(32) * 5
    y[0] = float("nan")
    module = _module(loss)

    value = module._compute_finite_loss(pred, loss.transform_target(y))
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(pred.grad).all()
    # Invalid samples must not contribute gradient.
    assert torch.all(pred.grad[0] == 0)


def test_all_nan_targets_give_zero_loss():
    module = _module("mse")
    pred = torch.randn(2, 1, 4, 4, requires_grad=True)
    y = torch.full((2, 4, 4), float("nan"))
    value = module._compute_finite_loss(pred, y)
    assert value.item() == 0.0
    value.backward()  # must not raise


def test_predict_shapes(loss_name):
    loss = get_loss(loss_name)
    pred = torch.randn(3, loss.n_outputs, 8, 8)
    out = loss.predict(pred)
    assert isinstance(out, dict) and out
    for key, val in out.items():
        assert val.shape[0] == 3
        if key != "surface_precip_quantiles":
            assert val.shape[1] == 1, key
    point = _module(loss)._point_estimate(pred)
    assert point.shape == (3, 8, 8)


def test_config_round_trip(loss_name):
    loss = get_loss(loss_name)
    rebuilt = get_loss(**loss.config())
    assert type(rebuilt) is type(loss)
    assert rebuilt.config() == loss.config()
    assert rebuilt.n_outputs == loss.n_outputs


def test_mse_matches_torch():
    pred, y = torch.randn(100), torch.randn(100)
    assert torch.allclose(get_loss("mse")(pred, y), nn.MSELoss()(pred, y))
    assert torch.allclose(get_loss("mae")(pred, y), nn.L1Loss()(pred, y))
    assert torch.allclose(get_loss("huber", delta=0.7)(pred, y), nn.HuberLoss(delta=0.7)(pred, y))


def test_logcosh_stable_and_correct():
    x = torch.tensor([0.0, 1.0, -1.0, 50.0, -200.0])
    value = get_loss("logcosh")(x, torch.zeros_like(x))
    expected = torch.stack([torch.log(torch.cosh(v)) if abs(v) < 20 else v.abs() - math.log(2) for v in x]).mean()
    assert torch.isfinite(value)
    assert torch.allclose(value, expected, atol=1e-4)


def test_weighted_mse_modes():
    pred = torch.zeros(4)
    y = torch.tensor([0.0, 0.05, 1.0, 10.0])
    plain = get_loss("mse")(pred, y)
    linear = get_loss("weighted_mse", alpha=1.0)(pred, y)
    step = get_loss("weighted_mse", alpha=1.0, mode="step", threshold=0.1)(pred, y)
    assert linear > plain and step > plain
    with pytest.raises(ValueError):
        get_loss("weighted_mse", mode="bogus")


def test_tweedie_minimised_at_target_mean():
    loss = get_loss("tweedie", p=1.5)
    y = torch.full((256,), 2.0)
    # The loss over raw outputs should be minimal when link(pred) == mean(y).
    best = torch.log(torch.expm1(torch.tensor(2.0)))  # softplus^-1(2)
    at_best = loss(torch.full((256,), best.item()), y)
    off = loss(torch.full((256,), best.item() + 1.0), y)
    assert at_best < off
    assert (loss.predict(torch.zeros(2, 1, 3, 3))["surface_precip"] > 0).all()
    with pytest.raises(ValueError):
        get_loss("tweedie", p=2.5)


def test_quantile_loss_and_median():
    loss = get_loss("quantile", quantiles=[0.1, 0.5, 0.9])
    assert loss.n_outputs == 3
    pred = torch.zeros(10, 3)
    y = torch.ones(10)
    # Pinball loss with target above prediction: q * err for each quantile.
    expected = torch.tensor([0.1, 0.5, 0.9]).mean()
    assert torch.allclose(loss(pred, y), expected)
    raw = torch.arange(3.0).view(1, 3, 1, 1)
    out = loss.predict(raw)
    assert out["surface_precip"].item() == 1.0  # median channel
    assert out["surface_precip_quantiles"].shape == (1, 3, 1, 1)


def test_detection_and_hybrid_targets():
    bce = get_loss("bce")
    y = torch.tensor([0.0, 0.05, 0.5, float("nan")])
    t = bce.transform_target(y)
    assert t[:3].tolist() == [0.0, 0.0, 1.0] and torch.isnan(t[3])
    assert "probability_of_precip" in bce.predict(torch.zeros(1, 1, 2, 2))

    hybrid = get_loss("hybrid", regression="huber", detection="focal", beta=0.5)
    assert hybrid.n_outputs == 2
    out = hybrid.predict(torch.zeros(2, 2, 4, 4))
    assert set(out) == {"surface_precip", "probability_of_precip"}
    rebuilt = get_loss(**hybrid.config())
    assert rebuilt.regression.name == "huber" and rebuilt.detection.name == "focal"


def test_torch_module_adapter_and_resolve():
    module = _module(nn.MSELoss())
    assert isinstance(module.criterion, TorchLossAdapter)
    assert module.criterion.name == "mse"
    assert resolve_loss(None).name == "mse"
    assert resolve_loss("huber", delta=2.0).delta == 2.0
    assert resolve_loss(get_loss("mae")).name == "mae"


def test_dotted_path_and_custom_registration():
    assert get_loss("satrain_models.losses.HuberLoss", delta=3.0).delta == 3.0

    @register_loss("_test_scaled_mse")
    class ScaledMSE(Loss):
        def __init__(self, scale: float = 2.0):
            super().__init__(scale=scale)
            self.scale = scale

        def forward(self, pred, target):
            return self.scale * ((pred - target) ** 2).mean()

    try:
        loss = get_loss("_test_scaled_mse", scale=4.0)
        assert loss.name == "_test_scaled_mse"
        assert loss.config() == {"name": "_test_scaled_mse", "args": {"scale": 4.0}}
        with pytest.raises(ValueError):
            register_loss("_test_scaled_mse")(type("Other", (Loss,), {}))
    finally:
        LOSSES.pop("_test_scaled_mse", None)

    with pytest.raises(ValueError):
        get_loss("does_not_exist")


def test_loss_toml(tmp_path):
    path = tmp_path / "huber.toml"
    path.write_text('name = "huber"\ntask = "regression"\n[args]\ndelta = 0.25\n')
    cfg = load_loss_toml(path)
    assert cfg == {"name": "huber", "args": {"delta": 0.25}}
    assert resolve_loss(path).delta == 0.25


def test_experiment_name_contains_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = SatRainEstimationModule(nn.Identity(), loss="quantile", approach="adamw", name="unet_gmi")
    assert module.experiment_name == "unet_gmi_quantile_adamw_v00"
    # Name is fixed once chosen, even if a checkpoint with that name appears.
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "unet_gmi_quantile_adamw_v00.ckpt").touch()
    assert module.experiment_name == "unet_gmi_quantile_adamw_v00"
    fresh = SatRainEstimationModule(nn.Identity(), loss="quantile", approach="adamw", name="unet_gmi")
    assert fresh.experiment_name == "unet_gmi_quantile_adamw_v01"


def test_compute_config_loss_fields():
    cfg = ComputeConfig(loss="huber", loss_args={"delta": 0.5}, accelerator="cpu")
    assert cfg.loss == "huber" and cfg.loss_args == {"delta": 0.5}
    assert ComputeConfig(accelerator="cpu").loss == "mse"
    assert cfg.to_dict()["loss_args"] == {"delta": 0.5}


def test_module_save_records_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from satrain_models.config import SatRainConfig
    from satrain_models.cli import load_saved_model

    config = SatRainConfig.from_dict(
        {
            "base_sensor": "gmi",
            "retrieval_input": [{"name": "gmi", "include_angles": False, "nan": -1.5}],
            "geometry": "on_swath",
            "subset": "xs",
            "format": "tabular",
        }
    )
    module = SatRainEstimationModule(nn.Linear(13, 5), loss="quantile", approach="adamw", name="mlp")
    path = module.save(config, tmp_path, model_config={"hidden_dims": [1]})
    saved = load_saved_model(path)
    assert saved["loss_config"]["name"] == "quantile"
    assert saved["model_config"] == {"hidden_dims": [1]}
    assert get_loss(**saved["loss_config"]).n_outputs == 5
