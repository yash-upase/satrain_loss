# satrain_loss — loss-function comparison on the SatRain benchmark

A fork of [ipwgml/satrain_models](https://github.com/ipwgml/satrain_models) in
which the PyTorch models are trained with a **selectable loss function** and the
losses are compared across models under one fixed recipe: seven estimation
losses (MSE, MAE, Huber, log-cosh, weighted MSE, Tweedie, quantile), training
subset `m`, 100 epochs. The study protocol, runner and leaderboard live in
[`loss_study/`](loss_study/README.md); [`loss_study/WALKTHROUGH.md`](loss_study/WALKTHROUGH.md)
walks through one complete run.

The Python package keeps its upstream name `satrain_models` so that the code
stays compatible with the reference repository.

## Overview

The ``satrain_models`` repository consists of two primary components:
    - The ``models`` directory provides specific implementations of machine-learning precipitation retrievals on the ``satrain`` dataset.
    - The ``satrain_models`` Python package provides shared utility functions and generic implementations of neural networks or other machine-learning models. The code in ``satrain_models`` is intended to be reused across model implementation in ``models`` or in external applications.


## Installation

Install the package in development mode:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e .[dev]
```

## Package Structure

```
satrain_models/
├── satrain_models/          # Python package with PyTorch models and utilities
│   ├── losses.py            # Loss registry (mse, mae, huber, logcosh, weighted_mse, tweedie, quantile, ...)
│   └── cli.py               # Shared --loss / --devices handling for train.py / test.py
├── models/                  # Model implementations
│   ├── model_name_1/        # Each model in its own folder
│   │   ├── requirements.txt # Conda-compatible requirements
│   │   └── ...              # Model-specific files
│   └── model_name_2/
│       ├── requirements.txt
│       └── ...
└── loss_study/              # Loss-function intercomparison: fixed configs, runner, aggregation
```

## Loss-function comparison study

The PyTorch models can be trained with any registered loss:

```bash
cd models/encoder_decoder
python train.py model_unet_medium.toml --loss huber --loss-arg delta=0.5
python test.py models/<experiment>.pt          # loss is read from the model file
```

The experiment name and all result files record the loss. `loss_study/`
contains the protocol and tooling for the collaborative study in which every model is
trained with every loss under one fixed recipe — see `loss_study/README.md`,
and `CONTRIBUTING.md` for adding a loss.

## Adding a Model

To add a new model implementation:

1. Create a new folder in `models/` with your model name
2. Add a `requirements.txt` file with conda-compatible dependencies
3. Implement ``train.py`` and ``test.py`` scripts that train the model and evaluate it on the testing data, respectively
4. Ensure the requirements can be installed with: `conda install --file requirements.txt`

## Model Requirements Format

Each model folder must contain a `requirements.txt` file that can be installed with conda. Example format:

```
numpy>=1.20.0
torch>=1.9.0
torchvision>=0.10.0
scikit-learn>=1.0.0
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Each contributor is free to organize their model folder as they see fit, as long as the `requirements.txt` requirement is met.
