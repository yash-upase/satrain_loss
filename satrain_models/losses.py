"""
satrain_models.losses
=====================

Loss functions for training SatRain retrievals and a small registry so that a
loss can be selected by name from a TOML file or the command line.

Every loss derives from :class:`Loss`, which extends ``torch.nn.Module`` with
the small amount of metadata the training and evaluation code needs to treat
the loss as a black box:

  - ``name``: short identifier used in experiment names and result attributes,
  - ``task``: ``"regression"``, ``"detection"`` or ``"hybrid"``,
  - ``n_outputs``: number of output channels the model must produce,
  - ``transform_target``: maps the raw surface-precipitation target to the
    quantity the loss is trained on (identity for regression),
  - ``predict``: maps raw model output to a dictionary of named retrieval
    outputs (``surface_precip``, ``probability_of_precip``, ...) understood by
    ``SatRainEstimationModule.get_retrieval_fn`` and the SatRain evaluator.

``forward(pred, target)`` receives tensors that have already been masked to
finite target values: ``pred`` has shape ``(N, n_outputs)`` (or ``(N,)`` when
``n_outputs == 1``) and ``target`` has shape ``(N,)``.

Adding a loss
-------------

.. code-block:: python

    from satrain_models.losses import Loss, register_loss

    @register_loss("my_loss")
    class MyLoss(Loss):
        def __init__(self, alpha: float = 1.0):
            super().__init__()
            self.alpha = alpha

        def forward(self, pred, target):
            return (self.alpha * (pred - target) ** 2).mean()

The loss is then available as ``get_loss("my_loss", alpha=2.0)``, from a TOML
file with ``name = "my_loss"``, or via ``--loss my_loss`` on the command line.
Losses defined outside this package can be referenced by dotted path, e.g.
``get_loss("my_package.my_module.MyLoss")``.
"""

import importlib
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


# Rain/no-rain threshold used by the SatRain evaluator (mm/h).
PRECIP_THRESHOLD = 0.1
HEAVY_PRECIP_THRESHOLD = 10.0


###############################################################################
# Base class and registry
###############################################################################


class Loss(nn.Module):
    """
    Base class for all SatRain training losses.

    Sub-classes must implement ``forward``. The remaining hooks have defaults
    suitable for a single-output regression loss.
    """

    #: Short identifier used in experiment names and result attributes.
    name: str = "loss"
    #: One of "regression", "detection", "hybrid".
    task: str = "regression"

    def __init__(self, **kwargs):
        super().__init__()
        # Constructor arguments are recorded so that the loss can be
        # re-instantiated from a saved model file.
        self._args: Dict[str, Any] = dict(kwargs)

    @property
    def n_outputs(self) -> int:
        """Number of output channels the model must produce."""
        return 1

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        """
        Map the raw surface-precipitation target to the quantity the loss is
        trained on. NaNs must be preserved so they can be masked out.
        """
        return target

    def predict(self, pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Map raw model output of shape ``(B, n_outputs, ...)`` to named retrieval
        outputs. Each value has shape ``(B, 1, ...)`` so that downstream code can
        uniformly select channel 0.
        """
        return {"surface_precip": pred[:, :1]}

    def config(self) -> Dict[str, Any]:
        """Serializable description sufficient to rebuild this loss."""
        return {"name": self.name, "args": dict(self._args)}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in self._args.items())


LOSSES: Dict[str, Type[Loss]] = {}


def register_loss(name: str) -> Callable[[Type[Loss]], Type[Loss]]:
    """Class decorator registering a :class:`Loss` under ``name``."""

    def decorator(cls: Type[Loss]) -> Type[Loss]:
        if name in LOSSES and LOSSES[name] is not cls:
            raise ValueError(f"A loss named '{name}' is already registered.")
        cls.name = name
        LOSSES[name] = cls
        return cls

    return decorator


def get_loss(name: str, args: Optional[Dict[str, Any]] = None, **kwargs) -> Loss:
    """
    Instantiate a loss by registered name or dotted import path.

    Args:
        name: Registered loss name (e.g. ``"huber"``) or a dotted path to a
            :class:`Loss` sub-class (e.g. ``"my_pkg.losses.MyLoss"``).
        args: Optional dictionary of constructor arguments (as stored by
            :meth:`Loss.config`).
        **kwargs: Additional constructor arguments; override ``args``.
    """
    cls = LOSSES.get(name)
    if cls is None:
        if "." not in name:
            raise ValueError(
                f"Unknown loss '{name}'. Registered losses: {sorted(LOSSES)}. "
                "Alternatively pass a dotted import path to a Loss sub-class."
            )
        module_name, _, cls_name = name.rpartition(".")
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        if not (isinstance(cls, type) and issubclass(cls, Loss)):
            raise TypeError(f"'{name}' does not refer to a satrain_models Loss.")
        if not getattr(cls, "name", None) or cls.name == Loss.name:
            cls.name = cls_name.lower()
    all_args = dict(args or {})
    all_args.update(kwargs)
    return cls(**all_args)


def load_loss_toml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a loss description TOML file.

    Expected layout::

        name = "huber"
        task = "regression"      # informational
        description = "..."      # informational

        [args]
        delta = 1.0

    Returns:
        A dictionary with keys ``name`` and ``args`` suitable for
        ``get_loss(**result)``.
    """
    path = Path(path)
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if "name" not in cfg:
        raise ValueError(f"Loss TOML file '{path}' is missing the 'name' key.")
    return {"name": cfg["name"], "args": dict(cfg.get("args", {}))}


def resolve_loss(spec: Union[str, Path, Loss, nn.Module, None] = None, **kwargs) -> Loss:
    """
    Turn a loss specification into a :class:`Loss` instance.

    ``spec`` may be ``None`` (MSE), a registered name, a dotted import path, a
    path to a TOML file, an existing :class:`Loss` or any ``nn.Module`` (which
    is wrapped by :class:`TorchLossAdapter`).
    """
    if spec is None:
        return get_loss("mse", **kwargs)
    if isinstance(spec, Loss):
        return spec
    if isinstance(spec, nn.Module):
        return TorchLossAdapter(spec)
    spec_str = str(spec)
    if spec_str.endswith(".toml"):
        cfg = load_loss_toml(spec_str)
        cfg["args"].update(kwargs)
        return get_loss(**cfg)
    return get_loss(spec_str, **kwargs)


###############################################################################
# Helpers
###############################################################################


def _as_flat(pred: torch.Tensor) -> torch.Tensor:
    """Drop a trailing singleton channel dimension: (N, 1) -> (N,)."""
    if pred.dim() == 2 and pred.shape[1] == 1:
        return pred[:, 0]
    return pred


###############################################################################
# Regression losses
###############################################################################


class TorchLossAdapter(Loss):
    """Wraps an arbitrary single-output ``nn.Module`` loss (e.g. ``nn.MSELoss``)."""

    task = "regression"

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.name = type(module).__name__.replace("Loss", "").lower() or "custom"

    def forward(self, pred, target):
        return self.module(_as_flat(pred), target)

    def config(self):
        return {"name": self.name, "args": {}}


@register_loss("mse")
class MSELoss(Loss):
    """Mean squared error (baseline)."""

    def forward(self, pred, target):
        return F.mse_loss(_as_flat(pred), target)


@register_loss("mae")
class MAELoss(Loss):
    """Mean absolute error (L1)."""

    def forward(self, pred, target):
        return F.l1_loss(_as_flat(pred), target)


@register_loss("huber")
class HuberLoss(Loss):
    """Huber loss: quadratic below ``delta``, linear above."""

    def __init__(self, delta: float = 1.0):
        super().__init__(delta=delta)
        self.delta = float(delta)

    def forward(self, pred, target):
        return F.huber_loss(_as_flat(pred), target, delta=self.delta)


@register_loss("logcosh")
class LogCoshLoss(Loss):
    """
    Log-cosh loss, ``mean(log(cosh(pred - target)))``, evaluated in the
    numerically stable form ``|x| + log1p(exp(-2|x|)) - log 2``.
    """

    def forward(self, pred, target):
        x = _as_flat(pred) - target
        ax = x.abs()
        return (ax + F.softplus(-2.0 * ax) - math.log(2.0)).mean()


@register_loss("weighted_mse")
class WeightedMSELoss(Loss):
    """
    MSE with per-sample weights that emphasise (heavy) precipitation.

    Args:
        alpha: Weight strength.
        threshold: Rain rate (mm/h) above which samples are up-weighted.
        mode: ``"linear"`` uses ``w = 1 + alpha * clamp(target, 0)``;
            ``"step"`` uses ``w = 1 + alpha * (target > threshold)``.
    """

    def __init__(self, alpha: float = 1.0, threshold: float = PRECIP_THRESHOLD, mode: str = "linear"):
        super().__init__(alpha=alpha, threshold=threshold, mode=mode)
        if mode not in ("linear", "step"):
            raise ValueError("'mode' must be 'linear' or 'step'.")
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.mode = mode

    def weights(self, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            return 1.0 + self.alpha * target.clamp(min=0.0)
        return 1.0 + self.alpha * (target > self.threshold).to(target.dtype)

    def forward(self, pred, target):
        w = self.weights(target)
        return (w * (_as_flat(pred) - target) ** 2).sum() / w.sum()


@register_loss("tweedie")
class TweedieLoss(Loss):
    """
    Tweedie deviance loss for zero-inflated, positive targets.

    The model output is passed through a softplus link so that the predicted
    mean ``mu`` is strictly positive; ``predict`` applies the same link. With
    ``1 < p < 2`` the Tweedie distribution is a compound Poisson-Gamma, which
    matches the point mass at zero rain and the heavy tail of rain rates.

    Args:
        p: Power parameter, must satisfy ``1 < p < 2``.
        eps: Small constant added to ``mu`` for numerical stability.
    """

    def __init__(self, p: float = 1.5, eps: float = 1e-6):
        super().__init__(p=p, eps=eps)
        if not 1.0 < p < 2.0:
            raise ValueError("Tweedie power 'p' must lie strictly between 1 and 2.")
        self.p = float(p)
        self.eps = float(eps)

    def link(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + self.eps

    def forward(self, pred, target):
        mu = self.link(_as_flat(pred).float())
        y = target.float().clamp(min=0.0)
        p = self.p
        dev = -y * mu.pow(1.0 - p) / (1.0 - p) + mu.pow(2.0 - p) / (2.0 - p)
        return dev.mean()

    def predict(self, pred):
        return {"surface_precip": self.link(pred[:, :1])}


@register_loss("quantile")
class QuantileLoss(Loss):
    """
    Pinball (quantile) loss over a set of quantiles.

    The model produces one channel per quantile. ``predict`` returns the
    median (or the quantile closest to 0.5) as ``surface_precip`` and the full
    set of quantiles as ``surface_precip_quantiles``.

    Args:
        quantiles: Increasing sequence of quantile levels in (0, 1).
    """

    def __init__(self, quantiles: Sequence[float] = (0.05, 0.16, 0.5, 0.84, 0.95)):
        quantiles = [float(q) for q in quantiles]
        super().__init__(quantiles=quantiles)
        if not quantiles or any(not 0.0 < q < 1.0 for q in quantiles):
            raise ValueError("'quantiles' must be a non-empty sequence of values in (0, 1).")
        self.register_buffer("quantiles", torch.tensor(quantiles), persistent=False)
        self.median_index = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))

    @property
    def n_outputs(self) -> int:
        return int(self.quantiles.numel())

    def forward(self, pred, target):
        if pred.dim() == 1:
            pred = pred[:, None]
        err = target[:, None] - pred
        q = self.quantiles.to(pred.dtype)
        return torch.maximum(q * err, (q - 1.0) * err).mean()

    def predict(self, pred):
        i = self.median_index
        return {
            "surface_precip": pred[:, i : i + 1],
            "surface_precip_quantiles": pred,
        }


###############################################################################
# Detection / hybrid reference implementations (phases 2 and 3)
###############################################################################


def _binary_target(target: torch.Tensor, threshold: float) -> torch.Tensor:
    invalid = torch.isnan(target)
    out = (target > threshold).to(target.dtype)
    out[invalid] = torch.nan
    return out


@register_loss("bce")
class BCELoss(Loss):
    """
    Binary cross-entropy on logits for rain / no-rain detection.

    Args:
        threshold: Rain rate (mm/h) defining the positive class.
        pos_weight: Optional weight for the positive class.
    """

    task = "detection"

    def __init__(self, threshold: float = PRECIP_THRESHOLD, pos_weight: Optional[float] = None):
        super().__init__(threshold=threshold, pos_weight=pos_weight)
        self.threshold = float(threshold)
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    def transform_target(self, target):
        return _binary_target(target, self.threshold)

    def forward(self, pred, target):
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=pred.device, dtype=pred.dtype)
        return F.binary_cross_entropy_with_logits(_as_flat(pred), target, pos_weight=pw)

    def predict(self, pred):
        return {"probability_of_precip": pred[:, :1]}


@register_loss("focal")
class FocalLoss(BCELoss):
    """
    Focal loss (Lin et al., 2017) on logits for rain / no-rain detection.

    Args:
        gamma: Focusing parameter.
        alpha: Positive-class balance in [0, 1]; ``None`` disables balancing.
        threshold: Rain rate (mm/h) defining the positive class.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25, threshold: float = PRECIP_THRESHOLD):
        Loss.__init__(self, gamma=gamma, alpha=alpha, threshold=threshold)
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.threshold = float(threshold)
        self.pos_weight = None

    def forward(self, pred, target):
        logits = _as_flat(pred)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        loss = bce * (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            loss = alpha_t * loss
        return loss.mean()


@register_loss("hybrid")
class HybridLoss(Loss):
    """
    Joint regression + detection loss with two output channels.

    Channel 0 is trained with a registered regression loss on the rain rate,
    channel 1 with a registered detection loss on the rain / no-rain flag:
    ``loss = regression + beta * detection``.

    Args:
        regression: Name of the regression loss (single output).
        detection: Name of the detection loss.
        beta: Weight of the detection term.
        regression_args / detection_args: Constructor arguments of the parts.
    """

    task = "hybrid"

    def __init__(
        self,
        regression: str = "mse",
        detection: str = "bce",
        beta: float = 1.0,
        regression_args: Optional[Dict[str, Any]] = None,
        detection_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            regression=regression,
            detection=detection,
            beta=beta,
            regression_args=dict(regression_args or {}),
            detection_args=dict(detection_args or {}),
        )
        self.regression = get_loss(regression, regression_args)
        self.detection = get_loss(detection, detection_args)
        if self.regression.n_outputs != 1 or self.detection.n_outputs != 1:
            raise ValueError("HybridLoss requires single-output regression and detection losses.")
        self.beta = float(beta)

    @property
    def n_outputs(self) -> int:
        return 2

    def forward(self, pred, target):
        reg = self.regression(pred[:, 0], self.regression.transform_target(target))
        det = self.detection(pred[:, 1], self.detection.transform_target(target))
        return reg + self.beta * det

    def predict(self, pred):
        out = dict(self.regression.predict(pred[:, 0:1]))
        out.update(self.detection.predict(pred[:, 1:2]))
        return out


__all__ = [
    "Loss",
    "LOSSES",
    "register_loss",
    "get_loss",
    "load_loss_toml",
    "resolve_loss",
    "TorchLossAdapter",
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "LogCoshLoss",
    "WeightedMSELoss",
    "TweedieLoss",
    "QuantileLoss",
    "BCELoss",
    "FocalLoss",
    "HybridLoss",
]
