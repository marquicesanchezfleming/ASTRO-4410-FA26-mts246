import glob
import numpy as np
from astropy.io import fits
from scipy import signal


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


def measure_signal_noise(data_list, bias_level, region=None):
    signal = []
    noise = []
    for img in data_list:
        if region is None:
            ny, nx = img.shape
            x0, x1 = nx // 4, 3 * nx // 4
            y0, y1 = ny // 4, 3 * ny // 4
        else:
            x0, x1, y0, y1 = region
        sub = img[y0:y1, x0:x1]
        signal.append(sub.mean() - bias_level)
        noise.append(sub.std())
    return np.array(signal), np.array(noise)


def subtract_read_noise(noise, read_noise_dn):
    variance = noise ** 2 - read_noise_dn ** 2
    return np.sqrt(np.clip(variance, 0, None))


def detect_regime(signal, noise, slope_target, tol=0.15, min_points=3, label=""):
    log_s = np.log10(signal)
    log_n = np.log10(noise)
    order = np.argsort(log_s)
    log_s, log_n = log_s[order], log_n[order]

    local_slope = np.gradient(log_n, log_s)
    mask_sorted = np.abs(local_slope - slope_target) < tol

    if mask_sorted.sum() < min_points:
        print(f"WARNING: only {mask_sorted.sum()} points matched slope "
              f"{slope_target}+/-{tol}{' (' + label + ')' if label else ''} -- "
              f"widen tol or inspect the plot manually.")

    mask = np.zeros_like(mask_sorted)
    mask[order] = mask_sorted
    return mask


def fit_gain(signal, noise, mask):
    log_s = np.log10(signal[mask])
    log_n = np.log10(noise[mask])
    offset = np.mean(log_n - 0.5 * log_s)  
    gain = 10 ** (2 * offset)
    return gain


def compute_fixed_pattern_component(signal, noise, read_noise_dn, gain):
    var_shot = gain * signal
    var_fp = noise ** 2 - read_noise_dn ** 2 - var_shot
    return np.sqrt(np.clip(var_fp, 0, None))


def fit_fixed_pattern(signal, fp_noise, mask):
    log_s = np.log10(signal[mask])
    log_n = np.log10(fp_noise[mask])
    offset = np.mean(log_n - log_s)  # forced-slope-1 fit
    fp_frac = 10 ** offset
    return fp_frac


def analyze_photon_transfer(frame_dir, bias_level, region=None,
                              read_noise_dn=None, exclude_keywords=(),
                              slope_target=0.5, tol=0.15):
    exptimes, data_list, files = load_series(frame_dir, exclude_keywords)
    signal, noise = measure_signal_noise(data_list, bias_level, region)

    order = np.argsort(signal)
    signal, noise, exptimes = signal[order], noise[order], exptimes[order]

    if read_noise_dn is not None:
        shot_noise = subtract_read_noise(noise, read_noise_dn)
    else:
        shot_noise = noise  

    read_noise_mask = detect_regime(signal, noise, slope_target=0.0, tol=tol,
                                     label="read noise floor")

    valid = shot_noise > 0
    shot_noise_mask = np.zeros_like(valid)
    shot_noise_mask[valid] = detect_regime(signal[valid], shot_noise[valid],
                                            slope_target, tol, label="shot noise")
    gain = fit_gain(signal, shot_noise, shot_noise_mask)

    fp_frac = None
    fp_noise = None
    fp_mask = None
    if read_noise_dn is not None:
        fp_noise = compute_fixed_pattern_component(signal, noise, read_noise_dn, gain)
        fp_valid = fp_noise > 0
        fp_mask = np.zeros_like(fp_valid)
        fp_mask[fp_valid] = detect_regime(signal[fp_valid], fp_noise[fp_valid],
                                           slope_target=1.0, tol=tol, label="fixed pattern")
        if fp_mask.sum() >= 2:
            fp_frac = fit_fixed_pattern(signal, fp_noise, fp_mask)

    print(f"\n{'exptime (s)':>12} {'signal (DN)':>14} {'noise (DN)':>12} "
          f"{'shot noise (DN)':>16} {'regime':>16}")
    for i, (t, s, n, sn) in enumerate(zip(exptimes, signal, noise, shot_noise)):
        if read_noise_mask[i]:
            regime = "read noise floor"
        elif shot_noise_mask[i]:
            regime = "shot noise"
        elif fp_mask is not None and fp_mask[i]:
            regime = "fixed pattern"
        else:
            regime = ""
        print(f"{t:12.3f} {s:14.1f} {n:12.2f} {sn:16.2f} {regime:>16}")

    print(f"\nFitted gain G = {gain:.4f} DN/electron")
    if read_noise_dn is not None:
        print(f"Read noise in electrons: {read_noise_dn * gain:.2f} e-")
    if fp_frac is not None:
        print(f"Fixed pattern noise fraction: {fp_frac*100:.3f}% of signal")
    elif read_noise_dn is not None:
        print("Fixed pattern regime not clearly resolved -- check the plot; "
              "may need more high-signal points or a wider tol.")

    return {
        "exptimes": exptimes,
        "signal": signal,
        "noise": noise,
        "shot_noise": shot_noise,
        "shot_noise_mask": shot_noise_mask,
        "read_noise_mask": read_noise_mask,
        "fp_noise": fp_noise,
        "fp_mask": fp_mask,
        "fp_frac": fp_frac,
        "gain": gain,
        "bias_level": bias_level,
        "read_noise_dn": read_noise_dn,
        "files": files,
    }


def plot_ptc(results):

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

    signal = results["signal"]
    noise = results["noise"]

    shot_mask = results["shot_noise_mask"]
    read_mask = results["read_noise_mask"]
    fp_mask = results["fp_mask"]

    gain = results["gain"]
    fp_frac = results["fp_frac"]
    read_noise_dn = results["read_noise_dn"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(signal, noise, 'x', color='lightgray', markersize=7, markeredgewidth=1.5, label=r"measured noise")
    ax.loglog(signal[read_mask], noise[read_mask], 'o', color='tab:red', markersize=7, label=r"read-noise regime")
    ax.loglog(signal[shot_mask], noise[shot_mask], 'o', color='tab:green', markersize=7, label=r"shot-noise regime")
    if fp_mask is not None and fp_mask.sum() > 0:
        ax.loglog(signal[fp_mask], noise[fp_mask], 'o', color='tab:orange', markersize=7, label=r"fixed-pattern regime")

    s_fit = np.logspace(np.log10(signal.min()), np.log10(signal.max()), 300)
    if read_noise_dn is not None:
        if fp_frac is not None:
            fp_term = (fp_frac * s_fit) ** 2
        else:
            fp_term = 0.0

        model = np.sqrt(read_noise_dn**2 + gain * s_fit + fp_term)
        ax.loglog(s_fit, model, '-', color='navy', linewidth=2.2,
            label=(r"$\sigma_{\rm tot}(S) = \sqrt{\sigma_{\rm read}^2 + GS+(f_{\rm FP}S)^2}$"))
        ax.axhline(read_noise_dn, color='tab:red', linestyle=':', linewidth=1.5,
            label=(rf"$\sigma_{{\rm read}} = {read_noise_dn:.3f}\,$DN"))

    ax.set_xlabel(r"Signal, bias-subtracted (DN)")
    ax.set_ylabel(r"Noise (DN)")
    ax.set_title(r"Photon Transfer Curve")
    ax.legend(fontsize=14, loc="upper left", frameon=True, fancybox=False, framealpha=0.9)
    fig.tight_layout()
    plt.savefig("/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 1 - CCD Characterization/plots/photon_transfer.png", dpi=1000, bbox_inches="tight")
    return fig