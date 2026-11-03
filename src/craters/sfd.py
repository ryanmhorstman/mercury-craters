"""Cumulative size-frequency distributions with Poisson error bars."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .geo import MERCURY_RADIUS_M  # noqa: E402


def box_area_km2(box, radius_m=MERCURY_RADIUS_M):
    lon_min, lon_max, lat_min, lat_max = box
    width_deg = (lon_max - lon_min) % 360 or 360
    r_km = radius_m / 1000.0
    return r_km**2 * np.radians(width_deg) * (np.sin(np.radians(lat_max)) - np.sin(np.radians(lat_min)))


def sqrt2_diameters(min_km, max_km):
    """Standard root-2 diameter steps (Crater Analysis Techniques Working Group, 1979)."""
    k = np.arange(np.floor(2 * np.log2(min_km)), np.ceil(2 * np.log2(max_km)) + 1)
    d = 2.0 ** (k / 2)
    return d[(d >= min_km * 0.999) & (d <= max_km * 1.001)]


def cumulative_sfd(diameters_km, area_km2, grid_km):
    """N(>= D) per km^2 at each grid diameter, with 1-sigma Poisson errors."""
    diameters_km = np.sort(np.asarray(diameters_km))
    counts = len(diameters_km) - np.searchsorted(diameters_km, grid_km, side="left")
    return counts / area_km2, np.sqrt(counts) / area_km2, counts


def plot_sfd(series, area_km2, grid_km, title, out_path):
    """series: {label: diameters_km}. Zero-count points are left off (log axes)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (label, diameters) in enumerate(series.items()):
        density, err, counts = cumulative_sfd(diameters, area_km2, grid_km)
        keep = counts > 0
        # Small multiplicative offsets keep overlapping error bars readable.
        x = grid_km[keep] * (1 + 0.025 * (i - (len(series) - 1) / 2))
        ax.errorbar(x, density[keep], yerr=err[keep], marker="o", ms=3.5, lw=1, capsize=2, label=f"{label} (n={len(diameters)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Diameter D (km)")
    ax.set_ylabel("Cumulative N(≥D) per km²")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
