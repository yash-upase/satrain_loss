"""Configuration classes for SatRain dataset and models."""

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# Try to import TOML library (tomllib in Python 3.11+, tomli for older versions)
try:
    import tomllib

    TOML_AVAILABLE = True
except ImportError:
    try:
        import tomli as tomllib

        TOML_AVAILABLE = True
    except ImportError:
        TOML_AVAILABLE = False
        tomllib = None

from satrain.input import InputConfig, calculate_input_features, parse_retrieval_inputs
from satrain.target import TargetConfig


@dataclass
class SatRainConfig:
    """Configuration for SatRain dataset including geometry, subset, retrieval input and target config.

    This class provides automatic parsing of retrieval inputs and target configurations
    from dictionaries when the satrain package is available.
    """

    base_sensor: str = "gmi"
    geometry: Optional[str] = None  # "gridded" or "on_swath"
    subset: Optional[str] = None  # "xs", "s", "m", "l", "xl"
    format: str = ("spatial",)
    retrieval_input: Optional[List[Union[str, Dict[str, Any]]]] = None
    target_config: Optional[Union[Dict[str, Any]]] = None
    hidden_layer: Optional[List[int]] = None

    def __init__(self, **kwargs):
        """Initialize with flexible keyword arguments from TOML."""
        self.base_sensor = kwargs.pop("base_sensor")
        self.geometry = kwargs.pop("geometry")
        self.subset = kwargs.pop("subset")
        self.format = kwargs.pop("format")
        self.retrieval_input = kwargs.pop("retrieval_input")
        self.target_config = kwargs.pop("target_config", {})
        self.hidden_layer = kwargs.pop("hidden_layer", [8, 2])

        if 0 < len(kwargs):
            raise ValueError(
                f"Encountered unsupported configuration arguments: {list(kwargs.keys())}"
            )

        # Call post-init processing
        self.__post_init__()

    def __post_init__(self):
        """Parse retrieval_input and target_config after initialization."""
        # Parse retrieval inputs if provided
        if self.retrieval_input is not None:
            self.retrieval_input = parse_retrieval_inputs(self.retrieval_input)

        # Parse target config if provided as dict
        if isinstance(self.target_config, dict):
            self.target_config = TargetConfig(**self.target_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SatRainConfig":
        """Create SatRainConfig from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            SatRainConfig instance
        """
        return cls(**config_dict)

    @classmethod
    def from_toml_file(cls, toml_path: Union[str, Path]) -> "SatRainConfig":
        """Create SatRainConfig from TOML file.

        Args:
            toml_path: Path to the TOML configuration file

        Returns:
            SatRainConfig instance

        Raises:
            ImportError: If TOML library is not available
            FileNotFoundError: If the TOML file doesn't exist
        """
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML library not available. Install with: pip install tomli"
            )

        toml_path = Path(toml_path)
        if not toml_path.exists():
            raise FileNotFoundError(f"TOML file not found: {toml_path}")

        with open(toml_path, "rb") as f:
            config_dict = tomllib.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_toml_string(cls, toml_string: str) -> "SatRainConfig":
        """Create SatRainConfig from TOML string.

        Args:
            toml_string: TOML configuration as string

        Returns:
            SatRainConfig instance

        Raises:
            ImportError: If TOML library is not available
        """
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML library not available. Install with: pip install tomli"
            )

        config_dict = tomllib.loads(toml_string)
        return cls.from_dict(config_dict)

    @property
    def tile_size(self) -> Tuple[int]:
        """Size of the tiles used for training."""
        if self.format == "spatial":
            return (256, 256)
        return (64, 64)

    @property
    def num_features(self) -> int:
        """Number of input features/channels."""
        return calculate_input_features(self.retrieval_input, stack=True)

    @property
    def features(self) -> int:
        """Names of the input features"""
        features = []
        for inpt in self.retrieval_input:
            features += list(inpt.features.keys())
        return features

    def to_dict(self) -> Dict[str, Any]:
        """Convert SatRainConfig to dictionary representation.

        Returns:
            Dictionary representation of the configuration
        """
        inpts = []
        for inpt in self.retrieval_input:
            dct = asdict(inpt)
            dct["name"] = inpt.__class__.__name__.lower()
            inpts.append(dct)

        dct = asdict(self)
        dct["retrieval_input"] = inpts
        return dct

    def to_toml_file(self, toml_path: Union[str, Path]) -> None:
        """Write SatRainConfig to TOML file.

        Args:
            toml_path: Path where to write the TOML configuration file

        Raises:
            ImportError: If TOML writing library is not available
        """
        try:
            import tomli_w
        except ImportError:
            raise ImportError(
                "TOML writing library not available. Install with: pip install tomli-w"
            )

        toml_path = Path(toml_path)
        config_dict = self.to_dict()

        with open(toml_path, "wb") as f:
            tomli_w.dump(config_dict, f)

    def get_experiment_name_prefix(self, model_type: str) -> str:
        """Generate experiment name prefix based on dataset configuration.

        Args:
            model_type: Type of model (e.g., "unet", "xgboost", "random_forest")

        Returns:
            String prefix for experiment naming
        """
        # Start with model type
        parts = [model_type]

        # Add sensor
        parts.append(self.base_sensor)

        # Add retrieval input names
        if self.retrieval_input:
            input_names = []
            for inpt in self.retrieval_input:
                if hasattr(inpt, "name"):
                    # Use name attribute (prioritized)
                    input_names.append(inpt.name)
                elif hasattr(inpt, "__class__"):
                    # Fallback to class name (lowercased)
                    input_names.append(inpt.__class__.__name__.lower())
            if input_names:
                parts.append("+".join(sorted(input_names)))

        # Add geometry
        if self.geometry:
            parts.append(self.geometry)

        # Add subset
        if self.subset:
            parts.append(self.subset)

        # Add format if not spatial (spatial is default)
        if self.format and self.format != "spatial":
            parts.append(self.format)

        return "_".join(parts)


@dataclass
class ComputeConfig(SatRainConfig):
    """Configutation of compute settings for a SatRain model."""

    approach: str = "adamw_warmup_cosine_annealing_restarts"
    learning_rate: Optional[float] = None

    # Loss selection: a registered name from ``satrain_models.losses`` (or a
    # dotted import path) and its constructor arguments (TOML ``[loss_args]``).
    loss: str = "mse"
    loss_args: Dict[str, Any] = field(default_factory=dict)

    max_epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    accelerator: str = "gpu"
    devices: List[int] = None
    precision: str = "float32"

    pin_memory: bool = True
    persistent_workers: bool = False
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 100
    check_val_every_n_epoch: int = 1

    model_confg: Optional[Dict[str, Any]] = None

    def __init__(self, **kwargs):
        """Parse compute configuration from keyword arguments."""
        self.approach = kwargs.pop("approach", "adamw_warmup_cosine_annealing_restarts")
        self.learning_rate = kwargs.pop("learning_rate", None)
        self.loss = kwargs.pop("loss", "mse")
        self.loss_args = dict(kwargs.pop("loss_args", {}) or {})
        self.max_epochs = kwargs.pop("max_epochs", 100)
        self.batch_size = kwargs.pop("batch_size", 8)
        self.num_workers = kwargs.pop("num_workers", 4)
        self.accelerator = kwargs.pop("accelerator", "cuda")
        self.devices = kwargs.pop("devices", 0)
        self.precision = kwargs.pop("precision", "float32")

        self.pin_memory = kwargs.pop("pin_memory", True)
        self.persistent_workers = kwargs.pop("persistent_workers", False)
        self.accumulate_grad_batches = kwargs.pop("accumulate_grad_batches", 1)
        self.log_every_n_steps = kwargs.pop("log_every_n_steps", 100)
        self.check_val_every_n_epoch = kwargs.pop("check_val_every_n_epoch", 1)
        # Debug knob: limit the number of train/val batches per epoch (smoke tests).
        self.limit_batches = kwargs.pop("limit_batches", None)
        self.model_config = kwargs.pop("model_config", {})

        if 0 < len(kwargs):
            raise ValueError(
                f"Encountered unsupported configuration arguments: {list(kwargs.keys())}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dictionary representation (used for result metadata)."""
        return {
            "approach": self.approach,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "loss_args": dict(self.loss_args),
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "accelerator": self.accelerator,
            "devices": self.devices,
            "precision": self.precision,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "log_every_n_steps": self.log_every_n_steps,
            "check_val_every_n_epoch": self.check_val_every_n_epoch,
            "limit_batches": self.limit_batches,
        }

    def trainer_overrides(self) -> Dict[str, Any]:
        """Extra ``lightning.Trainer`` keyword arguments derived from debug knobs."""
        if self.limit_batches is None:
            return {}
        return {
            "limit_train_batches": self.limit_batches,
            "limit_val_batches": self.limit_batches,
        }

    @property
    def device(self):
        """
        Helper function to calculate suitable single-device PyTorch device
        string from 'accelerator' and 'device' attributes.
        """
        if self.accelerator == "gpu":
            if isinstance(self.devices, list):
                return f"cuda:{self.devices[0]}"
            if isinstance(self.devices, int):
                return f"cuda:{self.devices}"
            return "cuda"
        return "cpu"

    @property
    def dtype(self):
        precision = self.precision
        if precision in ("bf16", "bfloat16"):
            return torch.bfloat16
        if precision in ("16", "16-true", "float16", "fp16"):
            return torch.float16
        if precision in ("32", "32-true", "float32", "fp32"):
            return torch.float32
        if precision in ("64", "64-true", "float64", "fp64"):
            return torch.float64
        if precision in ("16-mixed", "bf16-mixed"):
            return torch.bfloat16
        raise ValueError(f"Unsupported Lightning precision string: {precision}")
