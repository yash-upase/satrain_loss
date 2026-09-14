"""
satrain_models.cli
==================

Shared command-line helpers for the ``train.py`` / ``test.py`` scripts in
``models/``. They keep the loss-selection and override logic in one place so
that every model directory behaves the same way::

    python train.py model_medium.toml --loss huber --loss-arg delta=0.5 --devices 1
    python train.py model_medium.toml --loss ../../loss_study/losses/quantile.toml
    python test.py models/<experiment>.pt --devices 1

Precedence for the loss: ``--loss`` (with ``--loss-arg``) > ``compute.toml``
(``loss`` / ``[loss_args]``) > MSE.
"""

import argparse
import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from satrain_models.config import ComputeConfig, SatRainConfig
from satrain_models.losses import Loss, get_loss, load_loss_toml

LOGGER = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser, testing: bool = False) -> argparse.ArgumentParser:
    """
    Add the arguments shared by all training / testing scripts.

    Args:
        parser: The parser to extend.
        testing: If ``True``, omit training-only arguments (loss, dataset,
            max epochs); the loss is read from the saved model file instead.
    """
    if not testing:
        parser.add_argument(
            "--dataset-config",
            help="Path to dataset config file.",
            default="dataset.toml",
        )
        parser.add_argument(
            "--loss",
            default=None,
            help=(
                "Loss to train with: a registered name (mse, mae, huber, logcosh, "
                "weighted_mse, tweedie, quantile, ...), a dotted import path, or "
                "a path to a loss TOML file. Overrides 'loss' in compute.toml."
            ),
        )
        parser.add_argument(
            "--loss-arg",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Loss constructor argument (repeatable), e.g. --loss-arg delta=0.5",
        )
        parser.add_argument(
            "--max-epochs", type=int, default=None, help="Override max_epochs."
        )
        parser.add_argument(
            "--batch-size", type=int, default=None, help="Override batch_size."
        )
        parser.add_argument(
            "--limit-batches",
            type=int,
            default=None,
            help="Use only N train/val batches per epoch (smoke tests).",
        )
    parser.add_argument(
        "--precision", default=None, help="Override precision, e.g. '32' or 'bf16-mixed'."
    )
    parser.add_argument(
        "--compute-config",
        help="Path to compute config file.",
        default="compute.toml",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Override devices, e.g. '1' or '0,1'. Use 'cpu' to run on CPU.",
    )
    parser.add_argument(
        "--study",
        default=None,
        help="Optional study tag recorded in result files (e.g. loss_comparison).",
    )
    return parser


def _parse_value(text: str) -> Any:
    """Parse a KEY=VALUE value: numbers, lists, booleans, else string."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_loss_args(items: List[str]) -> Dict[str, Any]:
    """Parse repeated ``--loss-arg KEY=VALUE`` options."""
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--loss-arg expects KEY=VALUE, got '{item}'.")
        key, _, value = item.partition("=")
        out[key.strip()] = _parse_value(value.strip())
    return out


def load_compute_config(args: argparse.Namespace) -> ComputeConfig:
    """Load ``compute.toml`` (or ``--compute-config``) and apply CLI overrides."""
    path = Path(args.compute_config)
    if not path.exists():
        raise FileNotFoundError(f"Compute config not found: {path}")
    config = ComputeConfig.from_toml_file(path)
    LOGGER.info(f"Loaded compute config from: {path}")
    apply_overrides(config, args)
    return config


def load_dataset_config(args: argparse.Namespace) -> SatRainConfig:
    """Load the dataset TOML given by ``--dataset-config``."""
    path = Path(args.dataset_config)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    config = SatRainConfig.from_toml_file(path)
    LOGGER.info(f"Loaded SatRain config from: {path}")
    return config


def apply_overrides(config: ComputeConfig, args: argparse.Namespace) -> ComputeConfig:
    """Apply ``--devices`` / ``--max-epochs`` / ``--batch-size`` to a compute config."""
    devices = getattr(args, "devices", None)
    if devices is not None:
        if str(devices).lower() == "cpu":
            config.accelerator = "cpu"
            config.devices = 1
        else:
            config.devices = [int(d) for d in str(devices).split(",") if d.strip()]
            if config.accelerator == "cpu":
                config.accelerator = "cuda"
    if getattr(args, "max_epochs", None) is not None:
        config.max_epochs = args.max_epochs
    if getattr(args, "batch_size", None) is not None:
        config.batch_size = args.batch_size
    if getattr(args, "limit_batches", None) is not None:
        config.limit_batches = args.limit_batches
    if getattr(args, "precision", None) is not None:
        config.precision = args.precision
    return config


def resolve_loss_from_args(args: argparse.Namespace, compute_config: ComputeConfig) -> Loss:
    """
    Build the training loss from CLI arguments and the compute config.

    ``--loss`` wins over ``compute.toml``; ``--loss-arg`` values are merged on
    top of the arguments from either source.
    """
    cli_args = parse_loss_args(getattr(args, "loss_arg", []) or [])
    spec = getattr(args, "loss", None)
    if spec is None:
        name, loss_args = compute_config.loss, dict(compute_config.loss_args)
    elif str(spec).endswith(".toml"):
        cfg = load_loss_toml(spec)
        name, loss_args = cfg["name"], cfg["args"]
    else:
        name, loss_args = str(spec), {}
    loss_args.update(cli_args)
    loss = get_loss(name, loss_args)
    # Keep the compute config in sync so it can be recorded in metadata.
    compute_config.loss = loss.name
    compute_config.loss_args = dict(loss.config()["args"])
    LOGGER.info(f"Using loss: {loss}")
    return loss


def load_saved_model(path: Path) -> Dict[str, Any]:
    """
    Load a ``.pt`` / ``.ckpt`` file written by ``train.py`` and normalise it.

    Returns a dictionary with ``state_dict`` (model prefix stripped for
    Lightning checkpoints), ``satrain_config`` (dict), ``model_config`` (dict
    or ``None``) and ``loss_config`` (dict, defaults to MSE for legacy files).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Provided model path '{path}' doesn't exist.")
    if path.suffix == ".nc":
        raise ValueError(
            "You provided a NetCDF file; test.py expects a model file (.pt / .ckpt)."
        )
    try:
        loaded = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except Exception:  # pragma: no cover - legacy pickles
        loaded = torch.load(path, map_location=torch.device("cpu"), weights_only=False)

    if isinstance(loaded, dict) and "state_dict" in loaded:
        state = loaded["state_dict"]
    else:
        state = loaded
        loaded = {}

    if path.suffix == ".ckpt" or any(k.startswith("model.") for k in state):
        state = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state.items()
            if not k.startswith("criterion.")
        }

    loss_config = loaded.get("loss_config") or {"name": "mse", "args": {}}
    loss_config = {"name": loss_config.get("name", "mse"), "args": dict(loss_config.get("args", {}))}

    return {
        "state_dict": state,
        "satrain_config": loaded.get("satrain_config"),
        "model_config": loaded.get("model_config"),
        "loss_config": loss_config,
    }


def load_satrain_config_for_test(saved: Dict[str, Any], fallback: str = "dataset.toml") -> SatRainConfig:
    """SatRain config from a saved model file, falling back to a local TOML."""
    if saved.get("satrain_config") is not None:
        return SatRainConfig(**saved["satrain_config"])
    path = Path(fallback)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file has no 'satrain_config' and '{fallback}' not found."
        )
    LOGGER.info(f"Loaded SatRain config from {fallback} (fallback)")
    return SatRainConfig.from_toml_file(path)


DOMAINS = ("austria", "conus", "korea")


def run_evaluation(
    lightning_module,
    data_module,
    satrain_config: SatRainConfig,
    compute_config: ComputeConfig,
    domains=DOMAINS,
    batch_size: Optional[int] = None,
    tile_size=None,
    input_data_format: Optional[str] = None,
):
    """
    Evaluate a trained module on the SatRain test domains.

    Returns:
        An ``xarray.Dataset`` with a ``domain`` dimension holding the evaluator
        results for every domain.
    """
    import xarray as xr

    if batch_size is None:
        batch_size = compute_config.batch_size
    if input_data_format is None:
        input_data_format = "tabular" if satrain_config.format == "tabular" else "spatial"
    retrieval_fn = lightning_module.get_retrieval_fn(satrain_config, compute_config)

    results = []
    for domain in domains:
        LOGGER.info("Running evaluation for domain '%s'.", domain)
        evaluator = data_module.get_evaluator(domain)
        kwargs = {"batch_size": batch_size, "input_data_format": input_data_format}
        if input_data_format == "spatial":
            kwargs["tile_size"] = tile_size or satrain_config.tile_size
        evaluator.evaluate(retrieval_fn, **kwargs)
        results.append(evaluator.get_results())

    results = xr.concat(results, dim="domain")
    results["domain"] = list(domains)
    return results


def save_training_artifacts(
    lightning_module,
    log_dir,
    satrain_config: SatRainConfig,
    compute_config: ComputeConfig,
    model_family: str,
    model_config: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    study: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Path]:
    """
    Persist the outputs of a training run in the layout used across ``models/``:

    - ``netcdf_metrics/<experiment>.nc``: TensorBoard scalars with metadata,
    - ``models/<experiment>.pt``: weights + dataset / model / loss config.
    """
    from satrain_models.tensorboard_to_netcdf import tensorboard_to_netcdf

    experiment_name = lightning_module.experiment_name
    loss = lightning_module.criterion
    metadata = result_attrs(
        satrain_config,
        compute_config,
        loss,
        experiment_name,
        model_family,
        model_name=model_name,
        study=study,
        version=Path(log_dir).name,
        **extra,
    )
    # Legacy key retained for the existing notebooks.
    metadata["experiment"] = model_family

    netcdf_dir = Path("netcdf_metrics")
    netcdf_dir.mkdir(exist_ok=True)
    metrics_path = netcdf_dir / f"{experiment_name}.nc"
    try:
        tensorboard_to_netcdf(log_dir=log_dir, output_path=metrics_path, metadata=metadata)
    except Exception as exc:  # e.g. fast_dev_run without scalars
        LOGGER.warning(f"Could not export training metrics: {exc}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = lightning_module.save(satrain_config, models_dir, model_config=model_config)
    LOGGER.info(f"Model saved to: {model_path}")
    return {"metrics": metrics_path, "model": model_path}


def save_test_results(results, experiment_name: str, attrs: Dict[str, Any]) -> Path:
    """Write evaluator results to ``test_results/<experiment_name>.nc``."""
    output_path = Path("test_results")
    output_path.mkdir(exist_ok=True)
    results.attrs.update(attrs)
    path = output_path / f"{experiment_name}.nc"
    results.to_netcdf(path)
    LOGGER.info(f"Evaluation completed. Results saved to: {path}")
    return path


def result_attrs(
    satrain_config: SatRainConfig,
    compute_config: ComputeConfig,
    loss: Loss,
    experiment_name: str,
    model_family: str,
    model_name: Optional[str] = None,
    study: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Standard NetCDF attributes recorded with test results and training
    metrics so that runs can be grouped by model, loss and dataset.
    Values are coerced to NetCDF-compatible scalars / strings.
    """
    attrs: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_family": model_family,
        "model_name": model_name or "",
        "loss": loss.name,
        "loss_task": loss.task,
        "loss_args": repr(loss.config()["args"]),
        "n_outputs": loss.n_outputs,
        "base_sensor": satrain_config.base_sensor,
        "geometry": satrain_config.geometry or "",
        "subset": satrain_config.subset or "",
        "format": satrain_config.format,
        "approach": compute_config.approach,
        "max_epochs": compute_config.max_epochs,
        "batch_size": compute_config.batch_size,
        "precision": str(compute_config.precision),
    }
    if study:
        attrs["study"] = study
    for key, val in extra.items():
        if isinstance(val, bool):
            val = int(val)
        elif not isinstance(val, (int, float, str)):
            val = str(val)
        attrs[key] = val
    return attrs
