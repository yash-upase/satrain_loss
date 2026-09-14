#!/usr/bin/env python3
"""
Validate result files submitted to the loss-comparison study.

Checks that every given ``test_results/*.nc`` file

- carries the study tag and a known loss,
- was produced with the fixed study configuration (subset, approach, geometry),
- contains all three test domains with finite scalar metrics.

Usage::

    python loss_study/validate_results.py models/basic_unet/test_results/*.nc
    python loss_study/validate_results.py --changed   # files added/modified vs. main (CI)

Exit status is non-zero if any file fails.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

STUDY_DIR = Path(__file__).resolve().parent
REPO_DIR = STUDY_DIR.parent
STUDY_TAG = "loss_comparison"
REQUIRED_DOMAINS = {"austria", "conus", "korea"}
REQUIRED_METRICS = ("bias", "mae", "mse", "correlation_coef")


def _toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def expected_config() -> dict:
    compute = _toml(STUDY_DIR / "compute.toml")
    spatial = _toml(STUDY_DIR / "dataset_spatial.toml")
    tabular = _toml(STUDY_DIR / "dataset_tabular.toml")
    losses = {p.stem for p in (STUDY_DIR / "losses").glob("*.toml")}
    return {
        "approach": compute["approach"],
        "max_epochs": compute["max_epochs"],
        "subset": {"spatial": spatial["subset"], "tabular": tabular["subset"]},
        "geometry": {"spatial": spatial["geometry"], "tabular": tabular["geometry"]},
        "losses": losses,
    }


def validate(path: Path, expected: dict, strict: bool = True) -> list:
    problems = []
    try:
        ds = xr.load_dataset(path)
    except Exception as exc:
        return [f"cannot open: {exc}"]
    attrs = ds.attrs

    if attrs.get("study") != STUDY_TAG:
        problems.append(f"attr study={attrs.get('study')!r}, expected {STUDY_TAG!r} (run via loss_study/run_loss.py)")
    loss = attrs.get("loss")
    if loss is None:
        problems.append("missing 'loss' attribute")
    elif loss not in expected["losses"]:
        problems.append(f"loss {loss!r} has no loss_study/losses/{loss}.toml")

    fmt = attrs.get("format", "spatial")
    fmt = "tabular" if fmt == "tabular" else "spatial"
    for key in ("subset", "geometry"):
        want = expected[key][fmt]
        if attrs.get(key) != want:
            problems.append(f"attr {key}={attrs.get(key)!r}, study requires {want!r}")
    if attrs.get("approach") != expected["approach"]:
        problems.append(f"attr approach={attrs.get('approach')!r}, study requires {expected['approach']!r}")
    if strict and attrs.get("max_epochs") not in (None, expected["max_epochs"]):
        problems.append(f"attr max_epochs={attrs.get('max_epochs')}, study requires {expected['max_epochs']}")

    if "domain" not in ds.dims:
        problems.append("no 'domain' dimension")
    else:
        domains = set(map(str, ds["domain"].values))
        missing = REQUIRED_DOMAINS - domains
        if missing:
            problems.append(f"missing domains {sorted(missing)}")
        for metric in REQUIRED_METRICS:
            if metric not in ds:
                problems.append(f"missing metric '{metric}'")
            elif not np.all(np.isfinite(ds[metric].values)):
                problems.append(f"metric '{metric}' has non-finite values")
    return problems


def changed_files() -> list:
    base = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/main"], cwd=REPO_DIR, capture_output=True, text=True
    )
    ref = base.stdout.strip() or "HEAD~1"
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", ref, "--", "models/*/test_results/*.nc"],
        cwd=REPO_DIR, capture_output=True, text=True, check=True,
    )
    return [REPO_DIR / line for line in diff.stdout.split() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Result NetCDF files to check.")
    parser.add_argument("--changed", action="store_true", help="Check result files changed relative to origin/main.")
    parser.add_argument("--lenient", action="store_true", help="Do not enforce max_epochs.")
    args = parser.parse_args()

    files = [Path(f) for f in args.files]
    if args.changed:
        files += changed_files()
    if not files:
        print("No result files to validate.")
        return

    expected = expected_config()
    failed = 0
    for path in files:
        problems = validate(path, expected, strict=not args.lenient)
        status = "OK  " if not problems else "FAIL"
        print(f"{status} {path}")
        for p in problems:
            print(f"      - {p}")
        failed += bool(problems)

    print(f"\n{len(files) - failed}/{len(files)} files valid.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
