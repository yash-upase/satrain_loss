# Contributing

## Submitting loss-study results

Follow `loss_study/README.md`. In short: one branch `loss/<name>`, run
`python loss_study/run_loss.py <name>`, commit only the `*.nc` result files,
open a PR with the template.

## Adding a loss function

Losses live in `satrain_models/losses.py` (or any importable module) and are
selected by name, so no model script has to change.

```python
from satrain_models.losses import Loss, register_loss

@register_loss("scaled_mse")
class ScaledMSE(Loss):
    def __init__(self, scale: float = 1.0):
        super().__init__(scale=scale)      # record args so the loss can be rebuilt from a saved model
        self.scale = scale

    def forward(self, pred, target):       # pred: (N,) or (N, n_outputs); target: (N,); NaNs already removed
        return self.scale * ((pred - target) ** 2).mean()
```

Then add `loss_study/losses/scaled_mse.toml`:

```toml
name = "scaled_mse"
task = "regression"
description = "MSE multiplied by a constant."

[args]
scale = 2.0
```

and use it: `python train.py model_medium.toml --loss scaled_mse --loss-arg scale=3`.

Override the other hooks only when needed:

| hook | default | override when |
|---|---|---|
| `n_outputs` | `1` | the model must produce several channels (quantiles, mixture params, two heads) |
| `transform_target(y)` | identity | training on a derived quantity (rain flag for detection) — keep NaNs |
| `predict(raw)` | `{"surface_precip": raw[:, :1]}` | outputs need a link function (Tweedie) or a reduction (quantile median), or the loss trains a detector (`probability_of_precip` logits) or both (hybrid) |
| `task` | `"regression"` | `"detection"` / `"hybrid"`; used to pick the metrics reported by `aggregate.py` |

`predict` values must have shape `(B, 1, ...)`; the evaluator scores
`surface_precip` with regression metrics and `probability_of_precip`
(logits; sigmoid is applied for you) with detection metrics.

A loss in an external package is referenced by dotted path:
`--loss my_pkg.losses.MyLoss`.

Add a test in `tests/test_losses.py` if the loss has non-trivial maths; the
generic tests (masking, gradients, shapes, config round-trip) run automatically
for every registered loss.

## Adding a model

See `models/README.md`. Use `satrain_models.cli` helpers in `train.py` /
`test.py` so the model picks up `--loss`, `--devices`, `--study` and writes the
standard result attributes; pass `loss.n_outputs` as the number of output
channels and `loss=loss` to `SatRainEstimationModule`.

## Development

```bash
pip install -e .[dev]
pytest tests/ -q
```
