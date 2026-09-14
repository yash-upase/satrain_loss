"""
satrain_models.lightning
========================

Provides a LightningModule implementing three training recipes.
"""

import logging
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
    _LRScheduler,
)
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from satrain_models.losses import Loss, resolve_loss
from satrain_models.metrics import CorrelationCoef, PlotSamples, RelativeBias

# Import for type hints
if typing.TYPE_CHECKING:
    from .config import SatRainConfig

LOGGER = logging.getLogger(__name__)


APPROACHES = {
    "sgd_lr_search",
    "sgd",
    "sgd_reduce_on_plateau",
    "sgd_cosine_annealing_restarts",
    "sgd_warmup_cosine_annealing",
    "sgd_warmup_cosine_annealing_restarts",
    "adamw_lr_search",
    "adamw",
    "adamw_reduce_on_plateau",
    "adamw_cosine_annealing_restarts",
    "adamw_warmup_cosine_annealing",
    "adamw_warmup_cosine_annealing_restarts",
}


class SatRainEstimationModule(L.LightningModule):
    """
    Lightning module for training precipitation estimation models on the SatRain
    benchmark dataset.

    The module support a range of predefined training approaches that were assessed
    as part of the development of the SatRain baseline CNN retrieval.

    The supported training approaches are the following:

        - 'sgd_lr_search': Learning rate (LR) search with the Stochastic Gradient
          Descent (SGD) optimizer.
        - 'sgd': SGD without LR schedule.
        - 'sgd_reduce_on_plateau': SGD with LR reduction when validation loss plateaus.
        - 'sgd_cosine_annealing_restarts': SGD with cosine annealing LR schedule and warm restarts.
        - 'sgd_warmup_cosine_annealing': SGD with warmup and cosine annealing LR schedule.
        - 'sgd_warmup_cosine_annealing_restarts': SGD with warmup, cosine annealing LR schedule
          and warm restarts.
        - 'adamw_lr_search': LR search with the AdamW optimizer.
        - 'adamw': AdamW without schedule.
        - 'adamw_reduce_on_plateau': AdamW with LR reduction when validation loss plateaus.
        - 'adamw_cosine_annealing_restarts': AdamW cosine annealing LR schedule and warm restarts.
        - 'adamw_warmup_cosine_annealing': AdamW with warmup and cosine annealing LR schedule.
        - 'adamw_warmup_cosine_annealing_restarts': AdamW with warmup and cosine annealing LR schedule
          and warm restarts.

    """

    def __init__(
        self,
        model: nn.Module,
        loss: Union[Loss, nn.Module, str, None] = "mse",
        approach: str = "adamw_warmup_cosine_annealing",
        learning_rate: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            model: A PyTorch model implementing the model to train.
            loss: The loss used to train the model. Either a
                ``satrain_models.losses.Loss``, a registered loss name (e.g.
                ``"huber"``), a path to a loss TOML file, or any plain
                ``nn.Module`` loss (wrapped automatically). Defaults to MSE.
            approach: A string specifying the training approach.
            learning_rate: The initial learning rate to use.
            name: Optional name to use to prefix the experiment name with.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss"])  # keeps logs tidy

        self.model = model
        self.criterion: Loss = resolve_loss(loss)

        if approach not in APPROACHES:
            raise ValueError(f"Unknown approach '{approach}'. Choose from {APPROACHES}.")
        self.approach = approach
        self.learning_rate = learning_rate

        # Initialize validation metrics
        self.val_bias = RelativeBias()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_corrcoef = CorrelationCoef()
        self.plot_samples = PlotSamples()
        self.name = name

    @property
    def experiment_name(self):
        """
        Unique experiment name ``<name>_<loss>_<approach>_vNN``. The version is
        chosen on first access so that it doesn't exist in ``checkpoints/`` and
        is then fixed for the lifetime of the module.
        """
        cached = getattr(self, "_experiment_name", None)
        if cached is not None:
            return cached
        version = 0
        while True:
            parts = [] if self.name is None else [self.name]
            parts += [self.criterion.name, self.approach, f"v{version:02}"]
            name = "_".join(parts)
            if not (Path("checkpoints") / (name + ".ckpt")).exists():
                break
            version += 1
        self._experiment_name = name
        return name

    @experiment_name.setter
    def experiment_name(self, value: str):
        self._experiment_name = value

    @property
    def loss_name(self) -> str:
        """Name of the loss the module is trained with."""
        return self.criterion.name

    def loss_config(self) -> Dict[str, Any]:
        """Serializable description of the loss (see ``Loss.config``)."""
        return self.criterion.config()

    @staticmethod
    def _split_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract stacked input tensor and surface-precip target from a batch."""
        inputs, targets = batch
        # Handle input data (can be stacked tensor or dict)
        if isinstance(inputs, dict):
            x = torch.cat(list(inputs.values()), dim=1)
        else:
            x = inputs
        # Extract target (surface precipitation)
        if isinstance(targets, dict):
            y = targets["surface_precip"]
        else:
            y = targets
        return x, y.float()

    def _compute_finite_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss only over finite target values (no NaN or inf).

        Works for spatial ``(B, C, H, W)`` and tabular ``(B, C)`` predictions
        with targets of shape ``(B, H, W)`` / ``(B,)`` (a singleton channel
        dimension on the target is tolerated). The prediction channels are
        gathered channel-last so multi-output losses receive ``(N, C)``.

        Args:
            pred: Predictions tensor
            y: Target tensor (already transformed by the loss)

        Returns:
            Loss computed only over finite values, or zero if no finite values exist
        """
        if y.ndim == pred.ndim:
            y = y.select(1, 0)

        mask = torch.isfinite(y)
        # If no finite values, return zero loss (prevents NaN gradients)
        if not mask.any():
            return (pred * 0.0).sum()

        pred_masked = pred.movedim(1, -1)[mask]  # (N, C)
        if pred_masked.shape[-1] == 1:
            pred_masked = pred_masked[:, 0]
        return self.criterion(pred_masked, y[mask])

    def _point_estimate(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Reduce raw model output to the single channel used for validation
        metrics: the rain rate for regression / hybrid losses, the logit for
        detection losses. Returns shape ``(B, ...)`` without channel dimension.
        """
        outputs = self.criterion.predict(pred)
        for key in ("surface_precip", "probability_of_precip", "probability_of_heavy_precip"):
            if key in outputs:
                return outputs[key].select(1, 0)
        return next(iter(outputs.values())).select(1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Calculates loss for single input batch and logs errors.

        Args:
            batch: The input and target data.
            batch_idx: Not used.

        Returns:
            A torch.Tensor containing the loss.
        """
        x, y = self._split_batch(batch)
        y = self.criterion.transform_target(y)

        pred = self(x)

        # Calculate loss only over finite values
        loss = self._compute_finite_loss(pred, y)

        # Log loss
        self.log(
            f"train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        # Log LR (first param group) once per step for train
        opt = self.optimizers()
        if opt is not None:
            if isinstance(opt, (list, tuple)):
                lr = opt[0].param_groups[0]["lr"]
            else:
                lr = opt.param_groups[0]["lr"]
            self.log(
                "lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True
            )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step with comprehensive metrics computation."""
        x, y = self._split_batch(batch)
        y = self.criterion.transform_target(y)

        pred = self(x)

        # Compute main loss only over finite values
        loss = self._compute_finite_loss(pred, y)

        # Log main loss
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        # Validation metrics are computed on the loss' point estimate so they
        # are comparable across losses (e.g. quantile median, Tweedie mean).
        point = self._point_estimate(pred)
        if y.ndim == point.ndim + 1:
            y = y.select(1, 0)
        y_flat = y.reshape(-1)
        pred_flat = point.reshape(-1)

        # Remove NaN/inf values for metric computation (only use finite values)
        mask = torch.isfinite(y_flat) & torch.isfinite(pred_flat)
        if mask.sum() > 0:
            y_clean = y_flat[mask]
            pred_clean = pred_flat[mask]

            # Update metrics
            self.val_bias.update(pred_clean, y_clean)
            self.val_mse.update(pred_clean, y_clean)
            self.val_mae.update(pred_clean, y_clean)
            self.val_corrcoef.update(pred_clean, y_clean)

        # Only works for spatial format
        if point.ndim == 3:
            self.plot_samples.update(point.unsqueeze(1), y.unsqueeze(1))

        return loss

    def on_validation_epoch_end(self):
        self.log("val/bias", self.val_bias.compute())
        self.val_bias.reset()
        self.log("val/mae", self.val_mae.compute())
        self.val_mae.reset()
        self.log("val/mse", self.val_mse.compute())
        self.val_mse.reset()
        self.log("val/corrcoef", self.val_corrcoef.compute())
        self.val_corrcoef.reset()
        self.plot_samples.log(self)
        self.plot_samples.reset()

    def _steps(self, fraction: float) -> int:
        """
        Number of optimizer steps corresponding to ``fraction`` of the training
        run (at least 1, so tiny smoke-test runs don't break the schedulers).
        """
        return max(1, int(fraction * self.trainer.estimated_stepping_batches))

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:
        """
        Called by lightning to configure optimizers and schedulers.
        """
        hp = self.hparams
        lr = self.learning_rate

        # Learning rate search with SGD
        if hp.approach == "sgd_lr_search":
            LOGGER.info(
                "Performing learning rate search with SGD optimizer across %s steps.",
                self.trainer.estimated_stepping_batches,
            )
            lr = 1e-5
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = StepLR(
                optimizer,
                gamma=1e5 ** (1 / 100),
                step_size=1,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        # Learning rate search with AdamW
        if hp.approach == "adamw_lr_search":
            LOGGER.info(
                "Performing learning rate search with AdamW optimizer across %s steps.",
                self.trainer.estimated_stepping_batches,
            )
            lr = 1e-6
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = StepLR(
                optimizer,
                gamma=1e5 ** (1 / 100),
                step_size=1,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        # SGD, no LR schedule
        if hp.approach == "sgd":
            LOGGER.info("Running training with SGD optimizer, no LR schedule.")
            if lr is None:
                lr = 0.01
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            return optimizer

        # ADAM, no LR schedule
        if hp.approach == "adamw":
            LOGGER.info("Running training with SGD optimizer, no LR schedule.")
            if lr is None:
                lr = 1e-3
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            return optimizer

        # Learning rate search with AdamW
        if hp.approach == "adamw_lr_search":
            LOGGER.info(
                "Performing learning rate search with AdamW optimizer across %s steps.",
                self.trainer.estimated_stepping_batches,
            )
            lr = 1e-6
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = StepLR(
                optimizer,
                gamma=1e5 ** (1 / 100),
                step_size=1,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "sgd_reduce_on_plateau":
            if lr is None:
                lr = 0.01
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=5,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "adamw_reduce_on_plateau":
            if lr is None:
                lr = 1e-3
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=5,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "sgd_warmup_cosine_annealing":
            if lr is None:
                lr = 0.02
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=self._steps(0.1),
                    ),
                    CosineAnnealingLR(
                        optimizer,
                        T_max=self._steps(0.9),
                    ),
                ],
                milestones=[self._steps(0.1)],
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "adamw_warmup_cosine_annealing":
            if lr is None:
                lr = 4.14e-4
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
            scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=self._steps(0.1),
                    ),
                    CosineAnnealingLR(
                        optimizer,
                        T_max=self._steps(0.9),
                    ),
                ],
                milestones=[self._steps(0.1)],
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "sgd_warmup_cosine_annealing_restarts":
            if lr is None:
                lr = 0.02
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=self._steps(0.1),
                    ),
                    CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=self._steps(0.13),
                        T_mult=2,
                    ),
                ],
                milestones=[self._steps(0.1)],
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "adamw_warmup_cosine_annealing_restarts":
            if lr is None:
                lr = 4.14e-4
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=self._steps(0.1),
                    ),
                    CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=self._steps(0.13),
                        T_mult=2,
                    ),
                ],
                milestones=[self._steps(0.1)],
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "sgd_cosine_annealing_restarts":
            if lr is None:
                lr = 0.01
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                nesterov=True,
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self._steps(0.15),
                T_mult=2,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        if hp.approach == "adamw_cosine_annealing_restarts":
            if lr is None:
                lr = 1e-3
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self._steps(0.15),
                T_mult=2,
            )
            scheduler_conf = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val/loss",
            }
            return [optimizer], [scheduler_conf]

        raise ValueError("Unknow training approach '%s'.", hp.approach)

    def default_callbacks(self) -> List[L.Callback]:
        """
        Returns a sensible EarlyStopping and ModelCheckpoint list.
        Use in Trainer(callbacks=SatRainModule.default_callbacks(...))
        """
        callbacks = [
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="{epoch}-{val_loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                auto_insert_metric_name=False,
            ),
        ]
        callbacks[0].CHECKPOINT_NAME_LAST = self.experiment_name

        if self.approach in {"sgd_simple", "adamw_simple"}:
            callbacks += [
                EarlyStopping(
                    monitor="val/loss",
                    mode="min",
                    patience=10,
                    min_delta=0.0,
                    verbose=True,
                ),
            ]
        return callbacks

    def get_retrieval_fn(self, satrain_config, compute_config):
        """
        Get retrieval callback function for evaluation.
        """
        if compute_config.accelerator in ["gpu", "cuda"]:
            if compute_config.devices is None:
                device_ind = 0
            else:
                device_ind = compute_config.devices[0]
            device = f"cuda:{device_ind}"
        else:
            device = "cpu"
        dtype = compute_config.dtype
        self.model = self.model.to(device=device, dtype=dtype).eval()

        def retrieval_fn(input_data: xr.Dataset) -> xr.Dataset:
            """
            Run retrieval on input data.
            """
            feature_dim = 0
            if "scan" in input_data.dims:
                spatial_dims = ("scan", "pixel")
            elif "latitude" in input_data.dims:
                spatial_dims = ("latitude", "longitude")
            else:
                spatial_dims = ()

            if "batch" in input_data.dims:
                dims = ("batch",) + spatial_dims
                feature_dim += 1
            else:
                dims = spatial_dims

            inpt = {}
            for name in satrain_config.features:

                inpt_data = torch.tensor(input_data[name].data).to(device, dtype)
                if len(dims) == 1:
                    inpt_data = inpt_data.transpose(0, 1)
                inpt[name] = inpt_data

            inpt = torch.cat(list(inpt.values()), dim=feature_dim)

            with torch.no_grad():
                pred = self.model(inpt)
                if isinstance(pred, torch.Tensor):
                    # Let the loss turn raw channels into named outputs
                    # (e.g. quantile median, Tweedie link, detection logits).
                    if feature_dim == 0:
                        pred = {
                            key: val[0]
                            for key, val in self.criterion.predict(pred[None]).items()
                        }
                    else:
                        pred = self.criterion.predict(pred)

                results = xr.Dataset()
                if "surface_precip" in pred:
                    results["surface_precip"] = (
                        dims,
                        pred["surface_precip"]
                        .select(feature_dim, 0)
                        .float()
                        .cpu()
                        .numpy(),
                    )
                if "probability_of_precip" in pred:
                    pop = pred["probability_of_precip"].select(feature_dim, 0)
                    pop = torch.sigmoid(pop).float().cpu().numpy()
                    results["probability_of_precip"] = (dims, pop)
                    precip_flag = self.precip_threshold <= pop
                    results["precip_flag"] = (dims, precip_flag)

                if "probability_of_heavy_precip" in pred:
                    pohp = pred["probability_of_heavy_precip"].select(feature_dim, 0)
                    pohp = torch.sigmoid(pohp).float().cpu().numpy()
                    results["probability_of_heavy_precip"] = (dims, pohp)
                    heavy_precip_flag = self.heavy_precip_threshold <= pohp
                    results["heavy_precip_flag"] = (dims, heavy_precip_flag)

            return results

        return retrieval_fn

    #: Probability thresholds used to derive flags from detection outputs.
    precip_threshold: float = 0.5
    heavy_precip_threshold: float = 0.5

    def save(
        self,
        satrain_config: "SatRainConfig",
        output_path: Path,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model and configuration to path. The file name is determined using
        the experiment name of the lightning module.

        Args:
            satrain_config: The SatRain config the model was trained with.
            output_path: The path in in which to save the model.
            model_config: Optional dictionary describing the model architecture.

        Returns:
            The path of the saved file.
        """
        if not output_path.exists() or not output_path.is_dir():
            raise ValueError(
                "Output path for storing the model should point to an existing "
                "directory."
            )
        path = output_path / f"{self.experiment_name}.pt"
        state = self.model.state_dict()
        payload = {
            "state_dict": state,
            "satrain_config": satrain_config.to_dict(),
            "loss_config": self.loss_config(),
        }
        if model_config is not None:
            payload["model_config"] = model_config
        torch.save(payload, path)
        return path


class SatRainDetectionModule(SatRainEstimationModule):
    """
    Lightning module to train precipitation detection retrievals on the SatRain
    benchmark dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module = nn.BCEWithLogitsLoss(),
        approach: str = "adamw_warmup_cosine_annealing",
        learning_rate: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            model: A PyTorch model implementing the model to train.
            loss: A loss module defining the loss used to train the model.
            approach: A string specifying the training approach.
            learning_rate: The initial learning rate to use.
            name: Optional name to use to prefix the experiment name with.
        """
        super().__init__(
            model=model,
            loss=loss,
            approach=approach,
            learning_rate=learning_rate,
            name=name
        )

    @property
    def experiment_name(self):
        version = 0
        while True:
            if self.name is None:
                name = f"detection_{self.approach}_v{version:02}"
            else:
                name = f"{self.name}_detection_{self.approach}_v{version:02}"
            if not (Path("checkpoints") / (name + ".ckpt")).exists():
                break
            version += 1
        return name


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Calculates loss for single input batch and logs errors.

        Args:
            batch: The input and target data.
            batch_idx: Not used.

        Returns:
            A torch.Tensor containing the loss.
        """
        inputs, targets = batch

        # Handle input data (can be stacked tensor or dict)
        if isinstance(inputs, dict):
            # If not stacked, concatenate input channels
            x = torch.cat(list(inputs.values()), dim=1)
        else:
            x = inputs

        # Extract target (surface precipitation)
        if isinstance(targets, dict):
            y = targets["surface_precip"]
        else:
            y = targets

        invalid = torch.isnan(y)
        y = (0.1 < y).to(dtype=y.dtype)
        y[invalid] = torch.nan

        # Ensure target is float32 and has the right shape
        y = y.float()

        pred = self(x)

        # Squeeze output to match target shape if needed
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        # Calculate loss only over finite values
        loss = self._compute_finite_loss(pred, y)

        # Log loss
        self.log(
            f"train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        # Log LR (first param group) once per step for train
        opt = self.optimizers()
        if opt is not None:
            if isinstance(opt, (list, tuple)):
                lr = opt[0].param_groups[0]["lr"]
            else:
                lr = opt.param_groups[0]["lr"]
            self.log(
                "lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True
            )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step with comprehensive metrics computation."""
        inputs, targets = batch

        # Handle input data (can be stacked tensor or dict)
        if isinstance(inputs, dict):
            # If not stacked, concatenate input channels
            x = torch.cat(list(inputs.values()), dim=1)
        else:
            x = inputs

        # Extract target (surface precipitation)
        if isinstance(targets, dict):
            y = targets["surface_precip"]
        else:
            y = targets

        invalid = torch.isnan(y)
        y = (0.1 < y).to(dtype=y.dtype)
        y[invalid] = torch.nan

        # Ensure target is float32 and has the right shape
        y = y.float()

        pred = self(x)

        # Squeeze output to match target shape if needed
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        # Compute main loss only over finite values
        loss = self._compute_finite_loss(pred, y)

        # Log main loss
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        # Compute and log validation metrics
        # Flatten tensors for metric computation
        y_flat = y.view(-1)
        pred_flat = pred.view(-1)

        # Remove NaN/inf values for metric computation (only use finite values)
        mask = torch.isfinite(y_flat) * torch.isfinite(pred_flat)
        if mask.sum() > 0:
            y_clean = y_flat[mask]
            pred_clean = pred_flat[mask]

        # Only works for spatial format
        if pred.ndim == 4:
            self.plot_samples.update(pred, y)

        return loss


    def get_retrieval_fn(self, satrain_config, compute_config):
        """
        Get retrieval callback function for evaluation.
        """
        if compute_config.accelerator in ["gpu", "cuda"]:
            if compute_config.devices is None:
                device_ind = 0
            else:
                device_ind = compute_config.devices[0]
            device = f"cuda:{device_ind}"
        else:
            device = "cpu"
        dtype = compute_config.dtype
        self.model = self.model.to(device=device, dtype=dtype).eval()

        def retrieval_fn(input_data: xr.Dataset) -> xr.Dataset:
            """
            Run retrieval on input data.
            """
            feature_dim = 0
            if "scan" in input_data.dims:
                spatial_dims = ("scan", "pixel")
            elif "latitude" in input_data.dims:
                spatial_dims = ("latitude", "longitude")
            else:
                spatial_dims = ()

            if "batch" in input_data.dims:
                dims = ("batch",) + spatial_dims
                feature_dim += 1
            else:
                dims = spatial_dims

            inpt = {}
            for name in satrain_config.features:

                inpt_data = torch.tensor(input_data[name].data).to(device, dtype)
                if len(dims) == 1:
                    inpt_data = inpt_data.transpose(0, 1)
                inpt[name] = inpt_data

            inpt = torch.cat(list(inpt.values()), dim=feature_dim)

            with torch.no_grad():

                pred = self.model(inpt)
                if isinstance(pred, torch.Tensor):
                    pred = {"probability_of_precip": pred}

                results = xr.Dataset()
                if "surface_precip" in pred:
                    results["surface_precip"] = (
                        dims,
                        pred["surface_precip"]
                        .select(feature_dim, 0)
                        .float()
                        .cpu()
                        .numpy(),
                    )
                if "probability_of_precip" in pred:
                    pop = pred["probability_of_precip"].select(feature_dim, 0)
                    pop = torch.sigmoid(pop).float().cpu().numpy()
                    results["probability_of_precip"] = (dims, pop)
                    precip_flag = 0.5 <= pop
                    results["precip_flag"] = (dims, precip_flag)
                if "probability_of_heavy_precip" in pred:
                    pohp = pred["probability_of_heavy_precip"].select(feature_dim, 0)
                    pohp = torch.sigmoid(pohp).float().cpu().numpy()
                    results["probability_of_heavy_precip"] = (dims, pohp)
                    heavy_precip_flag = 0.5 <= pohp
                    results["heavy_precip_flag"] = (dims, heavy_precip_flag)

            return results

        return retrieval_fn
