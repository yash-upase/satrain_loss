# SatRain loss-comparison study

This directory holds everything needed to run and evaluate the **loss-function
intercomparison** on the SatRain benchmark: the PyTorch models in `models/`
are trained once per loss under an identical, fixed recipe, and the resulting
scores are collected into one leaderboard.

The study targets **regression of the surface rain rate** with these losses:

| loss | file | idea |
|---|---|---|
| `mse` (baseline) | `losses/mse.toml` | mean squared error |
| `mae` | `losses/mae.toml` | mean absolute error |
| `huber` | `losses/huber.toml` | MSE near zero, MAE for large errors |
| `logcosh` | `losses/logcosh.toml` | smooth MAE |
| `weighted_mse` | `losses/weighted_mse.toml` | MSE up-weighting rainy pixels |
| `tweedie` | `losses/tweedie.toml` | Tweedie deviance for zero-inflated rain rates |
| `quantile` | `losses/quantile.toml` | pinball loss, one output per quantile |

Detection and hybrid (rate + flag) losses will be added later as further
phases; the code already supports them.

**New here? Read [`WALKTHROUGH.md`](WALKTHROUGH.md)** — a step-by-step example
of running the `mae` loss from clone to pull request.
For the full background — what changed relative to upstream, diagrams of the
pipeline and the workflows — see
[`STUDY_GUIDE.md`](STUDY_GUIDE.md).

## Layout

```
loss_study/
├── README.md              this file (study protocol)
├── compute.toml           FIXED training recipe (optimizer schedule, epochs, precision)
├── dataset_spatial.toml   FIXED dataset for CNN models   (gmi / gridded / subset m)
├── dataset_tabular.toml   FIXED dataset for pixel models (gmi / on_swath / subset m)
├── models.toml            model matrix: which config + batch size per model dir
├── losses/<loss>.toml     one file per loss: name, args, description
├── run_loss.py            train + test ONE loss on all (or selected) models
├── validate_results.py    checks submitted result files (also run in CI)
├── aggregate.py           builds results/summary.csv + leaderboard.md
├── results/               generated tables (updated after each merge)
└── notebooks/compare_losses.ipynb   heat-maps and plots from summary.csv
```

Run outputs go where the repository already keeps them:

* `models/<model>/netcdf_metrics/<experiment>.nc` – training curves (**commit**)
* `models/<model>/test_results/<experiment>.nc` – evaluation scores (**commit**)
* `models/<model>/models/<experiment>.pt`, `checkpoints/`, `lightning_logs/` – weights/logs (**never commit**)

Experiment names always contain the loss, e.g.
`unet_gmi_gmi_gridded_m_huber_adamw_warmup_cosine_annealing_restarts_v00`,
and every result file carries `loss`, `loss_args`, `subset`, `approach`,
`study = "loss_comparison"` attributes so it can be grouped automatically.

## Running a loss

1. **Set up** (once)
   ```bash
   git clone https://github.com/yash-upase/satrain_loss.git && cd satrain_loss
   conda create -n satrain_loss python=3.11 && conda activate satrain_loss
   pip install git+https://github.com/ipwgml/satrain
   pip install -e .
   ```
   The SatRain data (subset `m`) is downloaded automatically on first use.

2. **Choose a loss**: one of the files in `losses/`. Its `[args]` may be
   tuned *before* the run (document why in the PR) — but the same args must be
   used for all models.

3. **Run** on a branch named `loss/<name>`:
   ```bash
   git checkout -b loss/huber
   python loss_study/run_loss.py huber --models basic_unet,ipwg_net,fully_connected --devices 0   # selected models
   python loss_study/run_loss.py huber --models ipwg_net --devices 0   # or one at a time
   python loss_study/run_loss.py huber --devices 0                     # every model in models.toml
   python loss_study/run_loss.py --list                                # show the model matrix
   ```
   `--devices` is the only thing that should change between machines. Do **not**
   edit `compute.toml`, `dataset_*.toml` or `models.toml`; if a model doesn't
   fit in memory, open an issue instead so that the batch size can be changed
   for everyone. Logs are in `loss_study/logs/`.

4. **Check** your results before submitting:
   ```bash
   python loss_study/validate_results.py models/*/test_results/*huber*.nc
   ```

5. **Submit** a pull request containing only
   * `models/*/test_results/*_<loss>_*.nc`
   * `models/*/netcdf_metrics/*_<loss>_*.nc`
   * (if changed) `loss_study/losses/<loss>.toml`

   and fill in the PR template. CI re-runs `validate_results.py` on the files you
   added. If a model failed to train, say so in the PR rather than omitting it silently.

### Smoke test first

Before a full run, make sure the pipeline works end-to-end for you:

```bash
python loss_study/run_loss.py huber --models basic_unet --subset xs --max-epochs 1 --limit-batches 2 --devices 0
# no GPU? add: --devices cpu --precision 32 --batch-size 2
```

Runs started with `--subset` / `--max-epochs` / `--limit-batches` /
`--batch-size` are tagged `loss_comparison_smoke` and ignored by
`aggregate.py` and rejected by `validate_results.py`, so they can't pollute
the study.

## Collecting and analysing results

* **Leaderboard:** after results are merged,
  ```bash
  python loss_study/aggregate.py          # -> results/summary.csv, leaderboard.md
  ```
  Only runs that match the current protocol (subset, epochs, optimizer
  schedule; see below) are included. `notebooks/compare_losses.ipynb` reads `results/summary.csv`.
* **Figures:** `plot_run.py` draws training curves, per-domain scores,
  spectral coherence and (with `--maps`) prediction maps and pred-vs-ref
  densities for any set of models and losses, e.g.
  ```bash
  python loss_study/plot_run.py --model basic_unet,ipwg_net,fully_connected --loss mse,huber --maps --devices 0
  ```
* **Later phases** (detection / hybrid losses): add `losses/<name>.toml` files
  and aggregate with `--task detection` / `--task hybrid`.

## Fixed protocol

| setting | value | file |
|---|---|---|
| sensor / inputs | GMI brightness temperatures, min-max normalised, no incidence angles | `dataset_*.toml` |
| training subset | `m` | `dataset_*.toml` |
| geometry | gridded (CNNs), on-swath tabular (MLP) | `dataset_*.toml` |
| optimizer / schedule | AdamW, warm-up + cosine annealing with restarts | `compute.toml` |
| epochs | **100** | `compute.toml` |
| precision | bf16-mixed | `compute.toml` |
| batch size | per model, identical across losses | `models.toml` |
| evaluation | SatRain test set, Austria / CONUS / Korea | `test.py` |

## Adding a new loss

See `CONTRIBUTING.md`. In short: subclass `satrain_models.losses.Loss`, decorate
with `@register_loss("name")`, add `losses/name.toml`. Multi-output losses set
`n_outputs` and/or override `predict`. No model script needs to change.

## Design notes (why it is built this way)

* **Loss is the only variable.** Architecture size, dataset subset, optimizer
  schedule, epochs and precision are pinned in this directory. Batch size is
  per model (memory), but identical across losses.
* **Losses own their output semantics.** `Loss.n_outputs` sizes the model head,
  `Loss.predict` turns raw channels into `surface_precip` /
  `probability_of_precip`, so the SatRain evaluator scores every loss with
  the same metrics regardless of how many channels the model produces.
* **Results are self-describing.** All grouping keys live in the NetCDF
  attributes; `aggregate.py` never parses filenames.
