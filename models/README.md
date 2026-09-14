# Models Directory

This directory contains individual model implementations for the SatRain dataset.

## Structure

Each model should be placed in its own subdirectory with the following requirements:

- **Folder name**: Use a descriptive name for your model (e.g., `basic_unet`)
- **requirements.txt**: Must be present and conda-compatible
- **``train.py`` and ``test.py``**: The training and testing functionality should be implemented in the
  Python scripts called ``train.py`` and ``test.py``, respectively. The ``test.py`` script should calculate the
  model accuracy on all testing domains.
  
## Dataset Configuration

The ``satrain_models`` package provides a ``SatRainConfig`` class that
encapsulates the configuration of the SatRain dataset. This configuration can be
loaded directly from a .toml file. Models are expected to use this mechanism so
they can seamlessly adapt to any input configuration. See the ``dataset.toml``
and ``train.py`` file in the ``basic_unet`` example model for an example how to
handle the dataset configuration.


## Utilities for PyTorch models

In addition, the satrain_models package offers a Lightning module
(``SatrainEstimationModule``) that implements a reference training procedure
along with essential logging. Reusing this module helps ensure that PyTorch
model training remains consistent and comparable. See the ``train.py`` script in
the ``basic_unet`` example for an example of how to use the lightning module.

## Loss selection and common command-line options

All PyTorch ``train.py`` / ``test.py`` scripts use ``satrain_models.cli`` and
therefore share these options:

```
--loss NAME|path.toml     loss to train with (default: 'loss' in compute.toml, else mse)
--loss-arg KEY=VALUE      loss constructor argument (repeatable)
--dataset-config PATH     dataset TOML (default dataset.toml)
--compute-config PATH     compute TOML (default compute.toml)
--devices 0,1 | cpu       override devices without editing compute.toml
--max-epochs N / --batch-size N
--study TAG               tag written to result files (used by loss_study/)
```

The loss determines the number of output channels (``loss.n_outputs``) and how
raw outputs are turned into retrieval quantities (``loss.predict``), so a
``train.py`` only needs to pass ``n_outputs=loss.n_outputs`` to its model
factory and ``loss=loss`` to the Lightning module. ``test.py`` reads the loss
back from the saved ``.pt`` file. Result files (``netcdf_metrics/``,
``test_results/``) carry ``loss``, ``loss_args``, ``subset``, ``approach`` and
``study`` attributes.



## Requirements File Format

The `requirements.txt` file should list all dependencies in a format that can be installed with:

```bash
conda install --file requirements.txt
```

Example `requirements.txt`:
```
numpy>=1.20.0
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
```
