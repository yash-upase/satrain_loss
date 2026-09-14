"""SatRain Models - ML model implementations for the SatRain dataset."""

from .config import ComputeConfig, SatRainConfig
from .datamodule import SatRainDataModule
from .losses import LOSSES, Loss, get_loss, load_loss_toml, register_loss, resolve_loss
from .lightning import SatRainEstimationModule, SatRainDetectionModule
from .random_forest import RandomForestRetrieval, create_random_forest_retrieval
from .resnet import ResNet, create_resnet, ResNetConfig, create_resnet_from_config  # veljko
from .tensorboard_to_netcdf import (
    extract_all_training_runs,
    extract_scalars_from_tensorboard,
    scalars_to_netcdf,
    tensorboard_to_netcdf,
)
from .unet import UNet, create_unet
from .fully_connected import FullyConnectedNetwork, create_fully_connected, FullyConnectedConfig
from .xgboost import XGBoostRetrieval, create_xgboost
from .efficient_net_v2 import create_efficient_net_v2, EfficientNetV2Config 
from .convnext import create_convnext, ConvNeXtConfig

__version__ = "0.1.0"
__author__ = "SatRain Contributors"

__all__ = [
    "UNet",
    "create_unet",
    "ResNet",  # veljko
    "create_resnet",  # veljko
    "ResNetConfig",  # veljko
    "create_resnet_from_config",  # veljko
    "FullyConnectedNetwork",
    "create_fully_connected",
    "FullyConnectedConfig",
    "XGBoostRetrieval",
    "create_xgboost",
    "RandomForestRetrieval",
    "create_random_forest_retrieval",
    "create_swinunet",
    "SatRainConfig",
    "ComputeConfig",  # veljko
    "SatRainEstimationModule",
    "SatRainDetectionModule",
    "SatRainDataModule",
    "Loss",
    "LOSSES",
    "get_loss",
    "register_loss",
    "resolve_loss",
    "load_loss_toml",
    "extract_scalars_from_tensorboard",
    "scalars_to_netcdf",
    "tensorboard_to_netcdf",
    "extract_all_training_runs",
    "create_efficient_net_v2",
    "EfficientNetV2Config",
    "create_convnext",
    "ConvNeXtConfig"
]
