#!/usr/bin/env python3
"""
Run one loss of the loss-comparison study across all models.

For every model listed in ``models.toml`` this script runs ``train.py`` and
then ``test.py`` inside the model directory with the fixed study configuration
(``compute.toml``, ``dataset_<format>.toml``, ``losses/<loss>.toml``).

Examples::

    # Full phase-1 run for the Huber loss on GPU 1
    python loss_study/run_loss.py huber --devices 1

    # Only two models, quick smoke test on the smallest subset
    python loss_study/run_loss.py quantile --models basic_unet,fully_connected \
        --subset xs --max-epochs 1 --devices cpu

    # Re-run evaluation only for models that already have weights
    python loss_study/run_loss.py mae --skip-train

Outputs land where the repository already keeps them:
``models/<model>/netcdf_metrics/*.nc``, ``models/<model>/models/*.pt`` (not
committed) and ``models/<model>/test_results/*.nc``. Logs are written to
``loss_study/logs/<loss>_<model>.log``.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

STUDY_DIR = Path(__file__).resolve().parent
REPO_DIR = STUDY_DIR.parent
MODELS_DIR = REPO_DIR / "models"
STUDY_TAG = "loss_comparison"


def load_matrix(only=None):
    with open(STUDY_DIR / "models.toml", "rb") as f:
        matrix = tomllib.load(f)
    if only:
        missing = [m for m in only if m not in matrix]
        if missing:
            raise SystemExit(f"Unknown model(s) {missing}. Known: {sorted(matrix)}")
        matrix = {k: matrix[k] for k in only}
    return matrix


def dataset_config(fmt: str, subset: str | None) -> Path:
    """Return the study dataset TOML, or a temporary copy with another subset."""
    base = STUDY_DIR / f"dataset_{fmt}.toml"
    if subset is None:
        return base
    tmp_dir = STUDY_DIR / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / f"dataset_{fmt}_{subset}.toml"
    text = base.read_text()
    text = re.sub(r'^subset\s*=\s*"[^"]*"', f'subset = "{subset}"', text, flags=re.M)
    path.write_text(text)
    return path


def newest_model_file(model_dir: Path, loss: str) -> Path | None:
    files = sorted(
        (model_dir / "models").glob(f"*_{loss}_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return files[-1] if files else None


def run(cmd, cwd: Path, log_path: Path, dry_run: bool) -> int:
    print(f"  $ (cd {cwd.relative_to(REPO_DIR)} && {' '.join(cmd)})")
    if dry_run:
        return 0
    env = dict(os.environ)
    # Make sure the scripts import *this* checkout of satrain_models.
    env["PYTHONPATH"] = str(REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("loss", help="Loss name; loss_study/losses/<loss>.toml must exist.")
    parser.add_argument("--models", default=None, help="Comma-separated subset of models.toml entries.")
    parser.add_argument("--devices", default=None, help="GPU index list like '0' or '0,1', or 'cpu'.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max_epochs (smoke tests only!).")
    parser.add_argument("--subset", default=None, help="Override dataset subset (smoke tests only!).")
    parser.add_argument("--limit-batches", type=int, default=None, help="Train/val batches per epoch (smoke tests only!).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the per-model batch size (smoke tests only!).")
    parser.add_argument("--precision", default=None, help="Override precision, e.g. '32' for CPU smoke tests.")
    parser.add_argument("--skip-train", action="store_true", help="Only run test.py on the newest matching model file.")
    parser.add_argument("--skip-test", action="store_true", help="Only run train.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    args = parser.parse_args()

    loss_toml = STUDY_DIR / "losses" / f"{args.loss}.toml"
    if not loss_toml.exists():
        raise SystemExit(f"No loss definition found at {loss_toml}")

    only = [m.strip() for m in args.models.split(",")] if args.models else None
    matrix = load_matrix(only)
    compute_toml = STUDY_DIR / "compute.toml"

    overrides = any(
        v is not None for v in (args.max_epochs, args.subset, args.limit_batches, args.batch_size)
    )
    study_tag = STUDY_TAG if not overrides else f"{STUDY_TAG}_smoke"
    if overrides:
        print("NOTE: --max-epochs/--subset/--limit-batches/--batch-size given; results are "
              "tagged as smoke runs and will be ignored by aggregate.py.")

    failures = []
    for name, spec in matrix.items():
        model_dir = MODELS_DIR / name
        log_path = STUDY_DIR / "logs" / f"{args.loss}_{name}.log"
        print(f"\n=== {name} / {args.loss} ===")
        dataset_toml = dataset_config(spec["dataset"], args.subset)

        common = ["--compute-config", str(compute_toml), "--study", study_tag]
        if args.devices:
            common += ["--devices", args.devices]
        if args.precision:
            common += ["--precision", args.precision]

        if not args.skip_train:
            cmd = [args.python, "train.py"]
            if spec.get("train_arg"):
                cmd.append(spec["train_arg"])
            cmd += [
                "--dataset-config", str(dataset_toml),
                "--loss", str(loss_toml),
                "--batch-size", str(args.batch_size or spec["batch_size"]),
            ] + common
            if args.max_epochs is not None:
                cmd += ["--max-epochs", str(args.max_epochs)]
            if args.limit_batches is not None:
                cmd += ["--limit-batches", str(args.limit_batches)]
            rc = run(cmd, model_dir, log_path, args.dry_run)
            if rc != 0:
                print(f"  TRAIN FAILED (exit {rc}); see {log_path}")
                failures.append((name, "train"))
                continue

        if not args.skip_test:
            model_file = None if args.dry_run else newest_model_file(model_dir, args.loss)
            if model_file is None and not args.dry_run:
                print(f"  no trained model file for loss '{args.loss}' in {model_dir / 'models'}")
                failures.append((name, "test"))
                continue
            cmd = [args.python, "test.py", str(model_file) if model_file else f"models/<newest *_{args.loss}_*.pt>"] + common
            rc = run(cmd, model_dir, log_path, args.dry_run)
            if rc != 0:
                print(f"  TEST FAILED (exit {rc}); see {log_path}")
                failures.append((name, "test"))

    print()
    if failures:
        print("Failed steps:", ", ".join(f"{m}:{s}" for m, s in failures))
        sys.exit(1)
    print(f"All done for loss '{args.loss}'. Next: python loss_study/aggregate.py")


if __name__ == "__main__":
    main()
