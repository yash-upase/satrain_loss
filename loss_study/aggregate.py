#!/usr/bin/env python3
"""
Collect the results of the loss-comparison study into tables.

Scans ``models/*/test_results/*.nc`` (evaluation scores) and
``models/*/netcdf_metrics/*.nc`` (training curves), keeps the files that carry
the study tag, and writes to ``loss_study/results/``:

- ``summary.csv``            long table: one row per (model, loss, domain, metric)
- ``training_summary.csv``   best validation metrics / epochs per (model, loss)
- ``leaderboard.md``         model x loss pivot tables per metric and domain,
                             plus mean rank of every loss across models

Usage::

    python loss_study/aggregate.py              # study runs only
    python loss_study/aggregate.py --all        # every result file with a 'loss' attr
    python loss_study/aggregate.py --task detection
"""

import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

import numpy as np
import pandas as pd
import xarray as xr

STUDY_DIR = Path(__file__).resolve().parent
REPO_DIR = STUDY_DIR.parent
MODELS_DIR = REPO_DIR / "models"
RESULTS_DIR = STUDY_DIR / "results"
STUDY_TAG = "loss_comparison"

# Metrics reported per task. Direction: -1 = lower is better, +1 = higher is better.
REGRESSION_METRICS = {
    "bias": -1,
    "mae": -1,
    "mse": -1,
    "smape": -1,
    "correlation_coef": +1,
}
DETECTION_METRICS = {
    "pod": +1,
    "far": -1,
    "hss": +1,
    "area_under_curve": +1,
}
HEAVY_METRICS = {
    "pod_heavy": +1,
    "far_heavy": -1,
    "hss_heavy": +1,
    "area_under_curve_heavy": +1,
}
DOMAIN_LABELS = {"austria": "Austria", "conus": "CONUS", "korea": "Korea"}


def _protocol() -> dict:
    """The fixed study settings a run must match to enter the leaderboard."""
    def load(name):
        with open(STUDY_DIR / name, "rb") as f:
            return tomllib.load(f)
    compute = load("compute.toml")
    return {
        "approach": compute["approach"],
        "max_epochs": compute["max_epochs"],
        "subsets": {load("dataset_spatial.toml")["subset"], load("dataset_tabular.toml")["subset"]},
    }


PROTOCOL = _protocol()


def _is_study_file(attrs: dict, include_all: bool) -> bool:
    """
    Study runs that match the current protocol (epochs, subset, approach).
    Runs made under an earlier protocol, or smoke runs, are left out unless
    --all is given.
    """
    if "loss" not in attrs:
        return False
    if include_all:
        return True
    if attrs.get("study") != STUDY_TAG:
        return False
    max_epochs = attrs.get("max_epochs")
    if max_epochs is not None and int(max_epochs) != int(PROTOCOL["max_epochs"]):
        return False
    if attrs.get("subset") and attrs.get("subset") not in PROTOCOL["subsets"]:
        return False
    if attrs.get("approach") and attrs.get("approach") != PROTOCOL["approach"]:
        return False
    return True


def collect_test_results(include_all: bool) -> pd.DataFrame:
    rows = []
    for path in sorted(MODELS_DIR.glob("*/test_results/**/*.nc")):
        try:
            ds = xr.load_dataset(path)
        except Exception as exc:  # pragma: no cover
            print(f"skip {path}: {exc}")
            continue
        attrs = dict(ds.attrs)
        if not _is_study_file(attrs, include_all):
            continue
        family = attrs.get("model_family") or path.parts[-3]
        for var in ds.data_vars:
            if ds[var].dims != ("domain",):
                continue
            for domain, value in zip(ds["domain"].values, ds[var].values):
                rows.append(
                    {
                        "model_family": family,
                        "model_dir": path.parts[-3],
                        "model_name": attrs.get("model_name", ""),
                        "loss": attrs["loss"],
                        "loss_task": attrs.get("loss_task", "regression"),
                        "loss_args": attrs.get("loss_args", ""),
                        "subset": attrs.get("subset", ""),
                        "approach": attrs.get("approach", ""),
                        "experiment_name": attrs.get("experiment_name", path.stem),
                        "num_parameters": attrs.get("num_parameters", np.nan),
                        "domain": str(domain),
                        "metric": var,
                        "value": float(value),
                        "file": str(path.relative_to(REPO_DIR)),
                    }
                )
    return pd.DataFrame(rows)


def collect_training_metrics(include_all: bool) -> pd.DataFrame:
    rows = []
    for path in sorted(MODELS_DIR.glob("*/netcdf_metrics/**/*.nc")):
        try:
            ds = xr.load_dataset(path)
        except Exception as exc:  # pragma: no cover
            print(f"skip {path}: {exc}")
            continue
        attrs = dict(ds.attrs)
        if not _is_study_file(attrs, include_all):
            continue
        row = {
            "model_family": attrs.get("model_family") or path.parts[-3],
            "model_dir": path.parts[-3],
            "loss": attrs["loss"],
            "experiment_name": attrs.get("experiment_name", path.stem),
            "file": str(path.relative_to(REPO_DIR)),
        }
        for var in ("val_loss", "val_mse", "val_mae", "val_corrcoef", "val_bias", "train_loss_epoch"):
            if var in ds:
                vals = ds[var].values
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    better = np.nanmax if var == "val_corrcoef" else np.nanmin
                    row[f"best_{var}"] = float(better(finite))
                    row[f"final_{var}"] = float(finite[-1])
                    row["n_val_epochs"] = int(finite.size) if var.startswith("val") else row.get("n_val_epochs")
        if "train_loss_step_wall_time" in ds:
            wt = ds["train_loss_step_wall_time"].values
            wt = wt[~pd.isnull(wt)]
            if wt.size > 1:
                row["train_hours"] = float((wt[-1] - wt[0]) / np.timedelta64(1, "h"))
        rows.append(row)
    return pd.DataFrame(rows)


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest run per (model_dir, loss, domain, metric)."""
    if df.empty:
        return df
    df = df.sort_values("experiment_name")
    return df.drop_duplicates(subset=["model_dir", "loss", "domain", "metric"], keep="last")


def _pivot(df: pd.DataFrame, metric: str, domain: str | None) -> pd.DataFrame:
    sub = df[df["metric"] == metric]
    if domain is not None:
        sub = sub[sub["domain"] == domain]
        table = sub.pivot_table(index="model_dir", columns="loss", values="value")
    else:
        table = sub.pivot_table(index="model_dir", columns="loss", values="value", aggfunc="mean")
    return table


def _md_table(table: pd.DataFrame, fmt: str = "{:.4g}") -> str:
    if table.empty:
        return "_no data_\n"
    cols = list(table.columns)
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in table.iterrows():
        cells = [
            (fmt.format(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
            if pd.notnull(v) else "—"
            for v in row.values
        ]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def leaderboard(df: pd.DataFrame, metrics: dict, title: str) -> str:
    out = [f"## {title}\n"]
    if df.empty:
        out.append("_no results found_\n")
        return "\n".join(out)
    rank_frames = []
    for metric, direction in metrics.items():
        if metric not in set(df["metric"]):
            continue
        out.append(f"### {metric} ({'lower' if direction < 0 else 'higher'} is better)\n")
        out.append("**Mean over domains**\n")
        table = _pivot(df, metric, None)
        out.append(_md_table(table))
        # Rank losses within each model row (1 = best), averaged over models.
        ranks = table.rank(axis=1, ascending=(direction < 0))
        rank_frames.append(ranks.mean(axis=0).rename(metric))
        for domain in sorted(df["domain"].unique()):
            out.append(f"<details><summary>{DOMAIN_LABELS.get(domain, domain)}</summary>\n")
            out.append(_md_table(_pivot(df, metric, domain)))
            out.append("</details>\n")
    if rank_frames:
        rank_table = pd.concat(rank_frames, axis=1)
        rank_table["mean_rank"] = rank_table.mean(axis=1)
        rank_table = rank_table.sort_values("mean_rank")
        out.append("### Mean rank of each loss across models (1 = best)\n")
        rt = rank_table.T
        rt.index.name = "model"
        out.append(_md_table(rt, fmt="{:.2f}"))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="Include every result file with a 'loss' attribute (also earlier protocols and smoke runs).")
    parser.add_argument("--task", default="regression", choices=["regression", "detection", "hybrid"],
                        help="Which loss task to report (default: regression).")
    parser.add_argument("--output", default=str(RESULTS_DIR), help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = collect_test_results(args.all)
    training = collect_training_metrics(args.all)

    if scores.empty:
        print("No test results found. Run loss_study/run_loss.py first.")
    else:
        scores = _dedupe(scores)
        scores.to_csv(out_dir / "summary.csv", index=False)
        print(f"wrote {out_dir / 'summary.csv'} ({len(scores)} rows, "
              f"{scores['loss'].nunique()} losses, {scores['model_dir'].nunique()} models)")
    if not training.empty:
        training.to_csv(out_dir / "training_summary.csv", index=False)
        print(f"wrote {out_dir / 'training_summary.csv'} ({len(training)} runs)")

    task_scores = scores[scores["loss_task"] == args.task] if not scores.empty else scores
    if args.task == "hybrid" and not scores.empty:
        task_scores = scores[scores["loss_task"].isin(["hybrid"])]

    sections = ["# Loss comparison leaderboard\n",
                f"Task: **{args.task}**. Source: `{'all result files' if args.all else 'study runs (study=' + STUDY_TAG + ')'}`. "
                f"Generated by `loss_study/aggregate.py`. "
                f"Protocol: subset {', '.join(sorted(PROTOCOL['subsets']))}, {PROTOCOL['max_epochs']} epochs, "
                f"`{PROTOCOL['approach']}`.\n"]
    if args.task == "regression":
        sections.append(leaderboard(task_scores, REGRESSION_METRICS, "Precipitation estimation"))
        sections.append(leaderboard(task_scores, DETECTION_METRICS, "Derived rain / no-rain detection (> 0.1 mm/h)"))
        sections.append(leaderboard(task_scores, HEAVY_METRICS, "Derived heavy-precipitation detection (> 10 mm/h)"))
    elif args.task == "detection":
        sections.append(leaderboard(task_scores, DETECTION_METRICS, "Rain / no-rain detection"))
    else:
        sections.append(leaderboard(task_scores, REGRESSION_METRICS, "Precipitation estimation head"))
        sections.append(leaderboard(task_scores, DETECTION_METRICS, "Detection head"))

    if not training.empty:
        sections.append("## Training summary\n")
        cols = [c for c in ["loss", "best_val_mse", "best_val_mae", "best_val_corrcoef",
                            "n_val_epochs", "train_hours", "experiment_name"] if c in training.columns]
        table = training.sort_values(["model_dir", "loss"]).set_index("model_dir")[cols]
        sections.append(_md_table(table, fmt="{}"))

    (out_dir / "leaderboard.md").write_text("\n".join(sections))
    print(f"wrote {out_dir / 'leaderboard.md'}")


if __name__ == "__main__":
    main()
