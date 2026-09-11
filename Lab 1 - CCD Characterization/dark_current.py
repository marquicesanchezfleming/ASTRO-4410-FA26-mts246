import glob
import numpy as np
from astropy.io import fits


def get_exptime(fits_path, keys=("EXPTIME", "EXPOSURE")):
    header = fits.getheader(fits_path)
    for key in keys:
        if key in header:
            return float(header[key])
    raise ValueError(f"No exposure time keyword found in {fits_path}")


def load_series(frame_dir, exclude_keywords=()):
    if not frame_dir.endswith("/"):
        frame_dir = frame_dir + "/"
    all_files = sorted(glob.glob(frame_dir + "*.fit*"))
    if not all_files:
        raise FileNotFoundError(f"No FITS files found in {frame_dir}")

    files = [f for f in all_files
             if not any(kw.lower() in f.lower() for kw in exclude_keywords)]

    exptimes = np.array([get_exptime(f) for f in files])
    data_list = [fits.getdata(f).astype(float) for f in files]

    order = np.argsort(exptimes)
    exptimes = exptimes[order]
    data_list = [data_list[i] for i in order]
    files = [files[i] for i in order]

    print(f"Loaded {len(files)} frames, exposure times (s): {exptimes}")
    return exptimes, data_list, files


def single_pixel_series(data_list, xy):
    x, y = xy
    return np.array([img[y, x] for img in data_list])


def region_median_series(data_list, region=None):
    levels = []
    for img in data_list:
        if region is None:
            sub = img
        else:
            x0, x1, y0, y1 = region
            sub = img[y0:y1, x0:x1]
        levels.append(np.median(sub))
    return np.array(levels)


def fit_dark_current(exptimes, levels, exclude_mask=None):
    mask = np.ones(len(exptimes), dtype=bool) if exclude_mask is None else ~exclude_mask
    slope, intercept = np.polyfit(exptimes[mask], levels[mask], 1)
    return slope, intercept


def analyze_dark_current(frame_dir, pixel_xy=None, region=None, exclude_keywords=()):
    exptimes, data_list, files = load_series(frame_dir, exclude_keywords)
    ny, nx = data_list[0].shape

    if pixel_xy is None:
        pixel_xy = (nx // 2, ny // 2)

    pixel_vals = single_pixel_series(data_list, pixel_xy)
    region_vals = region_median_series(data_list, region)

    pixel_slope, pixel_intercept = fit_dark_current(exptimes, pixel_vals)
    region_slope, region_intercept = fit_dark_current(exptimes, region_vals)

    print(f"\nSingle pixel {pixel_xy}: slope = {pixel_slope:.4f} DN/s, "
          f"intercept = {pixel_intercept:.1f} DN")
    print(f"Region median: slope = {region_slope:.4f} DN/s, "
          f"intercept = {region_intercept:.1f} DN")

    return {
        "exptimes": exptimes,
        "pixel_vals": pixel_vals,
        "region_vals": region_vals,
        "pixel_slope": pixel_slope,
        "pixel_intercept": pixel_intercept,
        "region_slope": region_slope,
        "region_intercept": region_intercept,
        "pixel_xy": pixel_xy,
        "region": region,
        "files": files,
    }


def plot_single_vs_region(results):
    import matplotlib.pyplot as plt

    exptimes = results["exptimes"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for ax, vals, slope, intercept, title in [
        (axes[0], results["pixel_vals"], results["pixel_slope"],
         results["pixel_intercept"], f"Single pixel {results['pixel_xy']}"),
        (axes[1], results["region_vals"], results["region_slope"],
         results["region_intercept"], "Region median"),
    ]:
        ax.plot(exptimes, vals, 'o', label="data")
        t_fit = np.linspace(0, exptimes.max(), 100)
        ax.plot(t_fit, slope * t_fit + intercept, '--', color='gray',
                 label=f"fit: {slope:.3f} DN/s")
        ax.set_xlabel("Exposure time (s)")
        ax.set_ylabel("DN")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    return fig


def plot_comparison(off_results, on_results):
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 16,
            "axes.linewidth": 1.5,
            "axes.unicode_minus": False,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
        })

    fig, ax = plt.subplots(figsize=(9, 6))
    for results, label, color in [(on_results, "Cooler ON", "#8236C7"), (off_results, "Cooler OFF", "#018943")]:
        t = results["exptimes"]
        v = results["region_vals"]
        slope = results["region_slope"]
        intercept = results["region_intercept"]
        ax.plot(t, v, 'o', color=color, label=f"{label} (measurements)")
        t_fit = np.linspace(0, t.max(), 100)
        ax.plot(t_fit, slope * t_fit + intercept, '--', color=color,
                 label=f"{label} fit: {slope:.4f} DN/s")

    ax.set_xlabel("Exposure time (s)")
    ax.set_ylabel("Region median (DN)")
    ax.legend()
    plt.savefig("/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 1 - CCD Characterization/plots/dark_current.png", dpi=1000, bbox_inches="tight")
    plt.savefig("/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 1 - CCD Characterization/nicer_plots/dark_current.pdf", bbox_inches="tight")
    return fig

def plot_comparison_baseline(off_results, on_results):
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-white")

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.linewidth": 1.5,
        "axes.unicode_minus": False,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.latex.preamble":
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{amsmath}"
            r"\usepackage{amssymb}",
    })

    fig, ax = plt.subplots(figsize=(9, 6))

    for results, label, color in [(on_results, "Cooler ON", "#8236C7"), (off_results, "Cooler OFF", "#018943")]:

        t = results["exptimes"]
        v = results["region_vals"]
        slope = results["region_slope"]
        intercept = results["region_intercept"]

        v_corrected = v - intercept
        ax.plot(t, v_corrected, 'o', color=color, label=f"{label} (measurements)")
        t_fit = np.linspace(0, t.max(), 200)
        ax.plot(t_fit, slope * t_fit, '--', color=color, label=f"{label} fit: {slope:.4f} DN/s")

    ax.axhline(0, color="0.7", linewidth=1)
    ax.set_xlabel("Exposure time (s)")
    ax.set_ylabel(r"Baseline-subtracted region median (DN)")
    ax.legend()
    plt.savefig("/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 1 - CCD Characterization/plots/dark_current_baseline_subtracted.png", dpi=1000, bbox_inches="tight")
    plt.savefig("/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 1 - CCD Characterization/nicer_plots/dark_current_baseline_subtracted.pdf", bbox_inches="tight")
    return fig