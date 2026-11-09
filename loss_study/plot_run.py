#!/usr/bin/env python3
"""
Publication-quality figures for one or more loss-study runs.

Reads the result files a run already produced (``netcdf_metrics/*.nc`` and
``test_results/*.nc``) and, optionally, the trained weights to draw prediction
maps. Pass several models and/or losses to compare them in the same figures;
every run keeps the same colour in every figure.

Figures (PNG, 300 dpi) written to ``--output`` (default
``loss_study/figures/``), ``<tag>`` = models and losses joined by ``+``:

- ``training_<tag>``   (a) train/val loss, (b) val MAE, (c) val correlation,
                       (d) learning rate, all vs. epoch; LR restarts shaded
- ``scores_<tag>``     bias, MAE, MSE, correlation per test domain
- ``coherence_<tag>``  spectral coherence vs. scale per domain (+ effective resolution)
- ``maps_<tag>``       (``--maps``) reference + every run for the wettest test
                       scene(s) of each domain, zoomed to the radar coverage,
                       with coastlines and borders
- ``density_<tag>``    (``--maps``) prediction vs. reference 2-D histograms for
                       the same scenes

Examples::

    python loss_study/plot_run.py --model basic_unet --loss mse
    python loss_study/plot_run.py --model basic_unet,ipwg_net,fully_connected --loss mse --maps --devices 0
    python loss_study/plot_run.py --model basic_unet --loss mse,mae,huber --maps --scenes 2 --devices 0

Scenes for the maps are chosen from the *reference* only (largest rain volume
among the first ``--search`` scenes), so selection is fast and all runs are
shown on identical scenes. All runs are evaluated on the gridded test scenes;
tabular models (fully_connected) are applied pixel-wise to the gridded inputs.
"""

import argparse
import glob
import struct
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

STUDY_DIR = Path(__file__).resolve().parent
REPO_DIR = STUDY_DIR.parent
MODELS_DIR = REPO_DIR / "models"
DOMAINS = ["austria", "conus", "korea"]
DOMAIN_LABELS = {"austria": "Austria", "conus": "CONUS", "korea": "Korea"}

# Okabe-Ito colour-blind-safe palette; one colour per run, fixed across figures.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# Discrete rain-rate levels (mm/h) for the maps.
RAIN_LEVELS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

NATURAL_EARTH = Path.home() / ".local/share/cartopy/shapefiles/natural_earth"

MODEL_LABELS = {
    "basic_unet": "U-Net",
    "encoder_decoder": "Encoder–decoder",
    "dense_encoder_decoder": "UNet++",
    "efficient_net_v2": "EfficientNetV2",
    "convnext": "ConvNeXt",
    "ResNet": "ResNet",
    "fully_connected": "MLP",
    "ipwg_net": "IPWG-Net",
}
LOSS_LABELS = {
    "mse": "MSE", "mae": "MAE", "huber": "Huber", "logcosh": "Log-cosh",
    "weighted_mse": "Weighted MSE", "tweedie": "Tweedie", "quantile": "Quantile",
}


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
    })


def save(fig, out: Path):
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def panel_label(ax, i, x=-0.12, y=1.04):
    ax.text(x, y, f"({chr(97 + i)})", transform=ax.transAxes, fontweight="bold", va="bottom")


# --------------------------------------------------------------------------- #
# Locating runs
# --------------------------------------------------------------------------- #


def newest(pattern: str):
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


def find_run(model: str, loss: str, subset: str):
    """Newest metrics / test-result / weight files for (model, loss)."""
    pat = f"*_{subset}_*{loss}_*.nc" if subset else f"*_{loss}_*.nc"
    d = MODELS_DIR / model
    run = {
        "model": model,
        "loss": loss,
        "metrics": newest(str(d / "netcdf_metrics" / pat)),
        "scores": newest(str(d / "test_results" / pat)),
        "weights": newest(str(d / "models" / pat.replace(".nc", ".pt"))),
    }
    # Guard against e.g. 'mse' matching 'weighted_mse' runs.
    for key in ("metrics", "scores", "weights"):
        f = run[key]
        if f is not None and f"_weighted_{loss}_" in f.name and loss != "weighted_mse":
            run[key] = None
    return run


def label_runs(runs):
    """Short legend labels: omit the part that is identical for all runs."""
    models = {r["model"] for r in runs}
    losses = {r["loss"] for r in runs}
    for i, r in enumerate(runs):
        m = MODEL_LABELS.get(r["model"], r["model"])
        l = LOSS_LABELS.get(r["loss"], r["loss"])
        if len(models) > 1 and len(losses) > 1:
            r["label"] = f"{m} · {l}"
        elif len(losses) > 1:
            r["label"] = l
        else:
            r["label"] = m
        r["color"] = PALETTE[i % len(PALETTE)]
    return runs


def _series(ds, var):
    if var not in ds:
        return None
    s = ds[var].dropna("step")
    return s if s.size else None


# --------------------------------------------------------------------------- #
# Training dynamics
# --------------------------------------------------------------------------- #


def _restart_epochs(ds):
    """Epochs at which the LR schedule restarts (large upward jumps in LR)."""
    lr, ep = _series(ds, "lr"), _series(ds, "epoch")
    if lr is None or ep is None:
        return []
    va = _series(ds, "val_loss")
    jumps = np.where(np.diff(lr.values) > 0.5 * np.nanmax(lr.values))[0]
    steps_per_epoch = (lr.step.values.max() + 1) / (va.size if va is not None else 1)
    return [float(s) / steps_per_epoch for s in lr.step.values[jumps + 1]]


def plot_training(runs, out):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    axs = axs.ravel()
    losses = {r["loss"] for r in runs}
    normalise = len(losses) > 1  # different losses have different units
    restarts = []
    for run in runs:
        if run["metrics"] is None:
            continue
        ds = xr.open_dataset(run["metrics"])
        c = run["color"]
        tr, va = _series(ds, "train_loss_epoch"), _series(ds, "val_loss")
        ref = float(va.values[0]) if (normalise and va is not None) else 1.0
        if tr is not None:
            axs[0].plot(np.arange(1, tr.size + 1), tr.values / ref, "--", color=c, lw=1.0, alpha=0.8)
        if va is not None:
            e = np.arange(1, va.size + 1)
            axs[0].plot(e, va.values / ref, "-", color=c, label=run["label"])
            best = int(np.nanargmin(va.values))
            axs[0].plot(e[best], va.values[best] / ref, "o", color=c, ms=4)
        for ax, var, better in [(axs[1], "val_mae", np.nanargmin), (axs[2], "val_corrcoef", np.nanargmax)]:
            s = _series(ds, var)
            if s is not None:
                e = np.arange(1, s.size + 1)
                ax.plot(e, s.values, color=c, label=run["label"])
                b = int(better(s.values))
                ax.plot(e[b], s.values[b], "o", color=c, ms=4)
        lr, va_ = _series(ds, "lr"), _series(ds, "val_loss")
        if lr is not None and va_ is not None and not axs[3].lines:
            # Fractional epoch from the optimizer step; the schedule is the same for all runs.
            steps_per_epoch = (lr.step.values.max() + 1) / va_.size
            axs[3].plot(lr.step.values / steps_per_epoch, lr.values * 1e4, color="0.25")
        if not restarts:
            restarts = _restart_epochs(ds)

    for ax in axs:
        for r in restarts:
            ax.axvline(r, color="0.6", lw=0.7, ls=":", zorder=0)
        ax.grid(alpha=0.25, lw=0.5)
    axs[0].set_ylabel("loss / loss$_{\\mathrm{val,\\,epoch\\,1}}$" if normalise else f"{LOSS_LABELS.get(next(iter(losses)), '')} loss")
    axs[0].set_title("Training (dashed) and validation (solid) loss")
    axs[1].set_ylabel("MAE (mm h$^{-1}$)")
    axs[1].set_title("Validation MAE")
    axs[2].set_ylabel("Pearson r")
    axs[2].set_title("Validation correlation")
    axs[3].set_ylabel("learning rate ($\\times 10^{-4}$)")
    axs[3].set_title("Learning-rate schedule (all runs)")
    for ax in axs[2:]:
        ax.set_xlabel("epoch")
    for i, ax in enumerate(axs):
        panel_label(ax, i, x=-0.16)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, -0.01, "Dotted lines: learning-rate restarts. Markers: best epoch.", ha="center",
             fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out)


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


def plot_scores(runs, out):
    metrics = [("bias", "Relative bias (%)"), ("mae", "MAE (mm h$^{-1}$)"),
               ("mse", "MSE (mm$^2$ h$^{-2}$)"), ("correlation_coef", "Pearson r")]
    valid = [r for r in runs if r["scores"] is not None]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axs = axs.ravel()
    n = max(len(valid), 1)
    width = 0.8 / n
    x = np.arange(len(DOMAINS))
    for i, run in enumerate(valid):
        ds = xr.open_dataset(run["scores"])
        offs = x - 0.4 + width * (i + 0.5)
        for ax, (var, _) in zip(axs, metrics):
            vals = np.array([float(ds[var].sel(domain=d)) for d in DOMAINS])
            bars = ax.bar(offs, vals, width * 0.92, color=run["color"], label=run["label"],
                          edgecolor="white", linewidth=0.4)
            if n <= 4:
                fmt = "%.0f" if var == "bias" else ("%.2f" if var == "correlation_coef" else "%.3g")
                ax.bar_label(bars, fmt=fmt, fontsize=6.5 if n > 2 else 7, padding=1.5,
                             rotation=90 if n > 2 else 0)
    for i, (ax, (var, title)) in enumerate(zip(axs, metrics)):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.margins(y=0.18)
        if var == "correlation_coef":
            ax.set_ylim(0, 1)
        panel_label(ax, i)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out)


# --------------------------------------------------------------------------- #
# Spectral coherence
# --------------------------------------------------------------------------- #


def plot_coherence(runs, out):
    doms = ["conus", "korea"]  # Austria is too small for the coherence estimate
    fig, axs = plt.subplots(1, len(doms), figsize=(7.2, 2.9), sharey=True)
    for run in runs:
        if run["scores"] is None:
            continue
        ds = xr.open_dataset(run["scores"])
        scales = np.where(np.isinf(ds.scales.values), np.nan, ds.scales.values)
        for ax, d in zip(axs, doms):
            coh = ds.spectral_coherence.sel(domain=d).values
            ax.semilogx(scales, coh, color=run["color"], label=run["label"])
            er = float(ds.effective_resolution.sel(domain=d))
            if np.isfinite(er):
                ax.axvline(er, color=run["color"], ls="--", lw=0.8)
    for i, (ax, d) in enumerate(zip(axs, doms)):
        ax.set_title(DOMAIN_LABELS[d])
        ax.set_xlabel("spatial scale (°)")
        ax.invert_xaxis()
        ax.grid(alpha=0.25, lw=0.5, which="both")
        panel_label(ax, i)
    axs[0].set_ylabel("spectral coherence")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.04, "Dashed: effective resolution. Austria omitted (domain too small).",
             ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out)


# --------------------------------------------------------------------------- #
# Coastlines / borders from Natural Earth shapefiles (no cartopy needed)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def _read_polylines(path: str):
    """Minimal reader for ESRI shapefiles of polylines/polygons -> list of (N, 2) arrays."""
    lines = []
    try:
        data = Path(path).read_bytes()
    except OSError:
        return lines
    pos = 100
    while pos + 8 <= len(data):
        _, length = struct.unpack(">2i", data[pos:pos + 8])
        rec = data[pos + 8: pos + 8 + 2 * length]
        pos += 8 + 2 * length
        if len(rec) < 44:
            continue
        shape_type = struct.unpack("<i", rec[:4])[0]
        if shape_type not in (3, 5):  # polyline, polygon
            continue
        bbox = struct.unpack("<4d", rec[4:36])
        n_parts, n_points = struct.unpack("<2i", rec[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", rec[44:44 + 4 * n_parts])) + [n_points]
        pts = np.frombuffer(rec, dtype="<f8", count=2 * n_points, offset=44 + 4 * n_parts).reshape(-1, 2)
        for a, b in zip(parts[:-1], parts[1:]):
            lines.append((bbox, pts[a:b]))
    return lines


def draw_geography(ax, extent):
    lon0, lon1, lat0, lat1 = extent
    for rel, lw in [("physical/ne_10m_coastline.shp", 0.6), ("cultural/ne_10m_admin_0_boundary_lines_land.shp", 0.45)]:
        for bbox, pts in _read_polylines(str(NATURAL_EARTH / rel)):
            if bbox[2] < lon0 or bbox[0] > lon1 or bbox[3] < lat0 or bbox[1] > lat1:
                continue
            ax.plot(pts[:, 0], pts[:, 1], color="0.15", lw=lw, ls="-" if "coast" in rel else "--", zorder=3)


# --------------------------------------------------------------------------- #
# Maps
# --------------------------------------------------------------------------- #


def _dataset_config_for_maps():
    from satrain_models import SatRainConfig
    return SatRainConfig.from_toml_file(STUDY_DIR / "dataset_spatial.toml")


def _dummy_retrieval(inp: xr.Dataset) -> xr.Dataset:
    dims = tuple(d for d in ("batch", "latitude", "longitude", "scan", "pixel", "samples") if d in inp.dims)
    shape = [inp.sizes[d] for d in dims]
    return xr.Dataset({"surface_precip": (dims, np.zeros(shape, "float32"))})


def select_scenes(evaluators, n_scenes: int, search: int):
    """Indices of the wettest scenes (by reference rain volume) per domain."""
    chosen = {}
    for domain, ev in evaluators.items():
        scored = []
        for index in range(search):
            try:
                res = ev.evaluate_scene(index, tile_size=(256, 256), overlap=0, batch_size=8,
                                        retrieval_fn=_dummy_retrieval, input_data_format="spatial", track=False)
            except (IndexError, ValueError):
                break
            ref = res["surface_precip_ref"].values
            scored.append((float(np.nansum(np.clip(ref, 0, None) * (ref >= 0.1))), index))
        scored.sort(reverse=True)
        chosen[domain] = [i for _, i in scored[:n_scenes]]
        print(f"  {DOMAIN_LABELS[domain]}: scenes {chosen[domain]}")
    return chosen


def predict_scenes(run, evaluators, chosen, devices):
    """Run one model on the chosen scenes -> {(domain, index): Dataset}."""
    from satrain_models import ComputeConfig, SatRainEstimationModule, get_loss
    from satrain_models.cli import load_satrain_config_for_test, load_saved_model

    saved = load_saved_model(run["weights"])
    satrain_config = load_satrain_config_for_test(saved)
    loss = get_loss(**saved["loss_config"])
    model = build_model(run["model"], saved, satrain_config, loss)
    model.load_state_dict(saved["state_dict"])
    compute = ComputeConfig(accelerator="cpu" if devices == "cpu" else "cuda",
                            devices=1 if devices == "cpu" else [int(devices)],
                            precision="32", batch_size=8)
    module = SatRainEstimationModule(model=model, loss=loss, approach=compute.approach)
    retrieval_fn = module.get_retrieval_fn(satrain_config, compute)
    fmt = "tabular" if satrain_config.format == "tabular" else "spatial"
    out = {}
    for domain, ev in evaluators.items():
        for index in chosen[domain]:
            out[(domain, index)] = ev.evaluate_scene(
                index, tile_size=None if fmt == "tabular" else satrain_config.tile_size, overlap=32,
                batch_size=4096 if fmt == "tabular" else 8, retrieval_fn=retrieval_fn,
                input_data_format=fmt, track=False)
    return out


SMALL_REFERENCE = 2000  # valid reference pixels below which the domain is a point-like network


def _zoom_extent(res, pad=0.5, min_span=4.0):
    """
    Map extent (lon0, lon1, lat0, lat1).

    - Small reference areas (Austria): the reference footprint, padded so the
      surrounding predicted field is visible.
    - Large areas (CONUS, Korea): the raining part of the reference (1st-99th
      percentile of rainy pixel positions), padded, at least `min_span` degrees,
      clipped to the reference coverage.
    """
    ref = res["surface_precip_ref"]
    lats, lons = ref.latitude.values, ref.longitude.values
    valid = np.isfinite(ref.values)
    if valid.sum() < SMALL_REFERENCE:
        r, c = np.where(valid)
        if r.size == 0:
            return float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())
        la, lo = lats[r], lons[c]
        return lo.min() - 1.0, lo.max() + 1.0, la.min() - 0.7, la.max() + 0.7
    r, c = np.where(valid & (ref.values >= 0.5))
    if r.size < 50:
        r, c = np.where(valid)
    vr, vc = np.where(valid)
    la, lo = lats[r], lons[c]
    lat0, lat1 = np.percentile(la, [1, 99]); lon0, lon1 = np.percentile(lo, [1, 99])
    lat0, lat1, lon0, lon1 = lat0 - pad, lat1 + pad, lon0 - pad, lon1 + pad
    for lo_, hi_, k in [(lat0, lat1, "lat"), (lon0, lon1, "lon")]:
        span = hi_ - lo_
        if span < min_span:
            mid = 0.5 * (lo_ + hi_)
            if k == "lat":
                lat0, lat1 = mid - min_span / 2, mid + min_span / 2
            else:
                lon0, lon1 = mid - min_span / 2, mid + min_span / 2
    vla, vlo = lats[vr], lons[vc]
    lat0, lat1 = max(lat0, vla.min() - pad), min(lat1, vla.max() + pad)
    lon0, lon1 = max(lon0, vlo.min() - pad), min(lon1, vlo.max() + pad)
    return lon0, lon1, lat0, lat1


def _deg_formatter(span):
    fmt = "{:.1f}°" if span < 4 else "{:.0f}°"
    return matplotlib.ticker.FuncFormatter(lambda v, _: fmt.format(v))


def _rain_cmap():
    base = plt.get_cmap("YlGnBu", len(RAIN_LEVELS) + 1)
    colors = ["#ececec"] + [base(i) for i in range(1, len(RAIN_LEVELS) + 1)]
    cmap = ListedColormap(colors[:-1])
    cmap.set_over(colors[-1])
    cmap.set_bad("white")
    norm = BoundaryNorm([0.0] + RAIN_LEVELS, cmap.N)
    return cmap, norm


def plot_maps(runs, preds, chosen, out_maps, out_density):
    cmap, norm = _rain_cmap()
    rows = [(d, i) for d in DOMAINS for i in chosen.get(d, [])]
    ncol = 1 + len(runs)
    any_res = next(iter(preds[runs[0]["label"]].values()))
    # Figure size from the zoomed aspect ratios.
    extents = {key: _zoom_extent(preds[runs[0]["label"]][key]) for key in rows}
    heights = []
    for key in rows:
        lon0, lon1, lat0, lat1 = extents[key]
        aspect = (lat1 - lat0) / ((lon1 - lon0) * np.cos(np.deg2rad(0.5 * (lat0 + lat1))))
        heights.append(aspect)
    col_w = 1.85
    fig_h = sum(col_w * h for h in heights) + 0.45 * len(rows) + 0.6
    fig, axs = plt.subplots(len(rows), ncol, figsize=(col_w * ncol + 0.6, fig_h), squeeze=False,
                            gridspec_kw={"height_ratios": heights})
    for r, key in enumerate(rows):
        domain, index = key
        ext = extents[key]
        ref_ds = preds[runs[0]["label"]][key]
        lats, lons = ref_ds.latitude.values, ref_ds.longitude.values
        dlat, dlon = abs(lats[1] - lats[0]), abs(lons[1] - lons[0])
        im_ext = (lons.min() - dlon / 2, lons.max() + dlon / 2, lats.min() - dlat / 2, lats.max() + dlat / 2)
        ref = ref_ds["surface_precip_ref"].values
        valid = np.isfinite(ref)
        small = valid.sum() < SMALL_REFERENCE
        # Small reference networks: show the full predicted field around them.
        fields = [("Reference", ref)] + [
            (run["label"], preds[run["label"]][key]["surface_precip"].values if small
             else np.where(valid, preds[run["label"]][key]["surface_precip"].values, np.nan))
            for run in runs]
        for c, (title, field) in enumerate(fields):
            ax = axs[r, c]
            shown = np.where(np.isfinite(field) & (field < RAIN_LEVELS[0]), 0.0, field)
            im = ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, norm=norm, extent=im_ext,
                           origin="upper" if lats[0] > lats[-1] else "lower",
                           interpolation="nearest", zorder=1)
            draw_geography(ax, ext)
            if small:
                ax.contour(lons, lats, valid.astype(float), levels=[0.5], colors="crimson",
                           linewidths=0.9, zorder=4)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (ext[2] + ext[3]))))
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.6)
            ax.tick_params(labelsize=6.5, length=2)
            ax.xaxis.set_major_formatter(_deg_formatter(ext[1] - ext[0]))
            ax.yaxis.set_major_formatter(_deg_formatter(ext[3] - ext[2]))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            if c > 0:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(title, fontweight="bold" if c == 0 else None)
            if c == 0:
                note = "\nred: reference network" if small else ""
                ax.set_ylabel(f"{DOMAIN_LABELS[domain]} (scene {index}){note}")
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", fraction=0.035,
                        pad=0.05 if len(rows) > 2 else 0.08, aspect=45, extend="max",
                        ticks=RAIN_LEVELS)
    cbar.set_label("surface precipitation (mm h$^{-1}$); grey: < 0.1, white: no reference")
    cbar.ax.set_xticklabels([f"{v:g}" for v in RAIN_LEVELS])
    save(fig, out_maps)

    # Density plots for the same scenes.
    fig, axs = plt.subplots(len(rows), len(runs), figsize=(2.15 * len(runs) + 0.8, 2.15 * len(rows)),
                            squeeze=False, sharex=True, sharey=True)
    bins = np.logspace(-1, 2, 41)
    h = None
    for r, key in enumerate(rows):
        ref = preds[runs[0]["label"]][key]["surface_precip_ref"].values
        for c, run in enumerate(runs):
            ax = axs[r, c]
            pred = preds[run["label"]][key]["surface_precip"].values
            m = np.isfinite(ref) & np.isfinite(pred) & ((ref >= 0.1) | (pred >= 0.1))
            x, y = np.clip(ref[m], 0.1, 100), np.clip(pred[m], 0.1, 100)
            if m.sum() < 200:
                ax.scatter(x, y, s=10, color=run["color"], edgecolor="k", linewidth=0.3, zorder=3)
                if m.sum() == 0:
                    ax.text(0.5, 0.5, "no rain at\nreference pixels", transform=ax.transAxes,
                            ha="center", va="center", color="0.4", fontsize=7)
            else:
                h = ax.hist2d(x, y, bins=[bins, bins], norm=LogNorm(), cmap="magma_r", cmin=1)
            ax.plot([0.1, 100], [0.1, 100], color="k", lw=0.7, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(0.1, 100); ax.set_ylim(0.1, 100)
            ax.set_aspect("equal")
            if m.sum() > 2:
                rr = np.corrcoef(ref[m], pred[m])[0, 1]
                bias = 100 * (np.nansum(pred[m]) - np.nansum(ref[m])) / max(np.nansum(ref[m]), 1e-6)
                ax.text(0.04, 0.96, f"r = {rr:.2f}\nbias = {bias:+.0f} %", transform=ax.transAxes,
                        va="top", fontsize=7)
            if r == 0:
                ax.set_title(run["label"])
            if r == len(rows) - 1:
                ax.set_xlabel("reference (mm h$^{-1}$)")
            if c == 0:
                ax.set_ylabel(f"{DOMAIN_LABELS[key[0]]}\nprediction (mm h$^{-1}$)")
    if h is not None:
        fig.colorbar(h[3], ax=axs.ravel().tolist(), label="pixel count", shrink=0.6)
    save(fig, out_density)


# --------------------------------------------------------------------------- #
# Model reconstruction
# --------------------------------------------------------------------------- #


def build_model(model_dir: str, saved, satrain_config, loss):
    """Recreate an untrained model matching a saved .pt (mirrors the test.py scripts)."""
    n_in, n_out = satrain_config.num_features, loss.n_outputs
    mc = saved.get("model_config") or {}
    if model_dir == "basic_unet":
        from satrain_models import create_unet
        return create_unet(n_channels=n_in, n_outputs=n_out, bilinear=mc.get("bilinear", True))
    if model_dir in ("encoder_decoder", "ipwg_net"):
        from satrain_models.encoder_decoder import BLOCK_FACTORIES, EncoderDecoder, EncoderDecoderConfig
        c = EncoderDecoderConfig(**mc)
        return EncoderDecoder(block_factory=BLOCK_FACTORIES[c.block_factory], in_channels=n_in,
                              channels=c.channels, depths=c.depths, out_channels=n_out, bilinear=c.bilinear)
    if model_dir == "dense_encoder_decoder":
        from satrain_models.encoder_decoder import EncoderDecoderConfig, BLOCK_FACTORIES
        from satrain_models.unetpp import DenseEncoderDecoder
        c = EncoderDecoderConfig(**mc)
        return DenseEncoderDecoder(block_factory=BLOCK_FACTORIES[c.block_factory], in_channels=n_in,
                                   channels=c.channels, depths=c.depths, out_channels=n_out)
    if model_dir == "efficient_net_v2":
        from satrain_models import create_efficient_net_v2
        return create_efficient_net_v2(n_channels=n_in, n_outputs=n_out,
                                       model_size=mc.get("model_size", "medium"), pretrained=False)
    if model_dir == "convnext":
        from satrain_models import create_convnext
        return create_convnext(n_channels=n_in, n_outputs=n_out,
                               model_size=mc.get("model_size", "small"), pretrained=False)
    if model_dir == "ResNet":
        from satrain_models import ResNetConfig, create_resnet_from_config
        c = ResNetConfig(**mc); c.n_outputs = n_out
        return create_resnet_from_config(c, num_features=n_in)
    if model_dir == "fully_connected":
        from satrain_models import create_fully_connected
        return create_fully_connected(input_dim=n_in, hidden_dims=mc.get("hidden_dims", [128, 128, 128]),
                                      output_dim=n_out, dropout=mc.get("dropout", 0.0))
    raise ValueError(f"Don't know how to build '{model_dir}'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model directory, or comma-separated list.")
    parser.add_argument("--loss", required=True, help="Loss name, or comma-separated list.")
    parser.add_argument("--subset", default="m", help="Training subset in the file names (default m; '' = any).")
    parser.add_argument("--maps", action="store_true", help="Also draw prediction maps (needs weights + test data).")
    parser.add_argument("--scenes", type=int, default=1, help="Scenes per domain for --maps.")
    parser.add_argument("--search", type=int, default=60, help="Pick the wettest of this many scenes per domain.")
    parser.add_argument("--devices", default="cpu", help="GPU index for --maps, or 'cpu'.")
    parser.add_argument("--output", default=str(STUDY_DIR / "figures"))
    args = parser.parse_args()

    set_style()
    runs = [find_run(m.strip(), l.strip(), args.subset)
            for m in args.model.split(",") for l in args.loss.split(",")]
    runs = label_runs(runs)
    for r in runs:
        print(f"{r['label']:24s} metrics={r['metrics'] and r['metrics'].name}  "
              f"scores={r['scores'] and r['scores'].name}")
    if not any(r["metrics"] or r["scores"] for r in runs):
        raise SystemExit("No result files found.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.replace(',', '+')}_{args.loss.replace(',', '+')}"
    plot_training(runs, out / f"training_{tag}")
    plot_scores(runs, out / f"scores_{tag}")
    plot_coherence(runs, out / f"coherence_{tag}")

    if args.maps:
        map_runs = [r for r in runs if r["weights"] is not None]
        for r in runs:
            if r["weights"] is None:
                print(f"no weights for {r['label']}; left out of the maps")
        if map_runs:
            from satrain_models import SatRainDataModule
            dm = SatRainDataModule(config=_dataset_config_for_maps(), num_workers=0)
            evaluators = {d: dm.get_evaluator(d) for d in DOMAINS}
            print("selecting scenes from the reference ...")
            chosen = select_scenes(evaluators, args.scenes, args.search)
            preds = {}
            for r in map_runs:
                print(f"predicting with {r['label']} ...")
                preds[r["label"]] = predict_scenes(r, evaluators, chosen, args.devices)
            plot_maps(map_runs, preds, chosen, out / f"maps_{tag}", out / f"density_{tag}")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
