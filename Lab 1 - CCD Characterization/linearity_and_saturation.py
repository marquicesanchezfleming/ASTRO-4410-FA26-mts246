import glob
import numpy as np
from astropy.io import fits

APERTURE_RADIUS_PX = 10      
BG_R_IN_PX = 15              
BG_R_OUT_PX = 25             
SATURATION_DN = 65000.0      
BIAS_LEVEL_DN = None         
SHORT_EXPTIME_CUTOFF_S = 0.1 

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


def find_centroid(image, box_half_size=15, guess=None):
    if guess is None:
        cy0, cx0 = np.unravel_index(np.argmax(image), image.shape)
    else:
        cx0, cy0 = guess

    y0, y1 = max(0, cy0 - box_half_size), cy0 + box_half_size
    x0, x1 = max(0, cx0 - box_half_size), cx0 + box_half_size
    cutout = image[y0:y1, x0:x1]

    yy, xx = np.mgrid[y0:y1, x0:x1]
    total = cutout.sum()
    cx = (xx * cutout).sum() / total
    cy = (yy * cutout).sum() / total

    print(f"Centroid: ({cx:.2f}, {cy:.2f})")
    return cx, cy


def aperture_sum(image, cx, cy, radius):
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return image[mask].sum(), mask.sum()  


def annulus_background(image, cx, cy, r_in, r_out):
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)
    return np.median(image[mask])


def analyze_linearity(frame_dir, aperture_radius=APERTURE_RADIUS_PX,
                       bg_r_in=BG_R_IN_PX, bg_r_out=BG_R_OUT_PX,
                       bias_level=BIAS_LEVEL_DN, saturation_dn=SATURATION_DN,
                       linear_max_frac=0.7,
                       short_exptime_cutoff_s=SHORT_EXPTIME_CUTOFF_S):
    """
    short_exptime_cutoff_s: frames below this exposure time are
        flagged as unreliable (shutter/reset timing) and excluded from
        the fit. If your whole series falls below the default 0.1s
        (e.g. a very bright source saturating in milliseconds),
        lowering this is a deliberate, caveated choice -- state it
        explicitly in your writeup, and check the residuals of the
        resulting fit to see whether any of the shortest exposures
        still stand out as genuinely anomalous.
    """
    exptimes, data_list, files = load_series(frame_dir)
    mid_idx = len(data_list) // 2
    cx, cy = find_centroid(data_list[mid_idx])

    peak_vals = []
    aperture_vals = []
    for img in data_list:
        peak_vals.append(img.max())
        bg = annulus_background(img, cx, cy, bg_r_in, bg_r_out)
        raw_sum, n_pix = aperture_sum(img, cx, cy, aperture_radius)
        bg_subtracted = raw_sum - bg * n_pix
        if bias_level is not None:
            bg_subtracted -= bias_level * n_pix
        aperture_vals.append(bg_subtracted)

    peak_vals = np.array(peak_vals)
    aperture_vals = np.array(aperture_vals)

    short_exptime_flag = exptimes < short_exptime_cutoff_s

    saturated_flag = peak_vals >= saturation_dn
    near_saturated_flag = peak_vals >= linear_max_frac * saturation_dn

    fit_mask = (~saturated_flag) & (~near_saturated_flag) & (~short_exptime_flag)
    if fit_mask.sum() < 2:
        print("WARNING: fewer than 2 points in the linear regime -- "
              "widen linear_max_frac, lower short_exptime_cutoff_s, "
              "or check your exposure series.")
        slope = intercept = None
    else:
        slope, intercept = np.polyfit(exptimes[fit_mask], aperture_vals[fit_mask], 1)
        print(f"Linear fit (using {fit_mask.sum()} points): "
              f"signal = {slope:.2f} * exptime + {intercept:.2f}")

        # Residuals for the fitted points -- check whether any point
        # still stands out despite passing the cutoff, especially
        # relevant if short_exptime_cutoff_s was lowered from default.
        pred = slope * exptimes[fit_mask] + intercept
        resid_pct = (aperture_vals[fit_mask] - pred) / pred * 100
        print("Residuals (%) for fitted points:")
        for t, r in zip(exptimes[fit_mask], resid_pct):
            print(f"    {t:8.4f}s : {r:+.2f}%")

    print(f"\n{'exptime (s)':>12} {'peak (DN)':>12} {'aperture sum (DN)':>20} {'flag':>12}")
    for t, p, a, sat, near, short in zip(exptimes, peak_vals, aperture_vals,
                                          saturated_flag, near_saturated_flag,
                                          short_exptime_flag):
        flag = "SATURATED" if sat else ("near-sat" if near else ("short-exp" if short else ""))
        print(f"{t:12.3f} {p:12.1f} {a:20.1f} {flag:>12}")

    return {
        "exptimes": exptimes,
        "peak_vals": peak_vals,
        "aperture_vals": aperture_vals,
        "centroid": (cx, cy),
        "saturated_flag": saturated_flag,
        "near_saturated_flag": near_saturated_flag,
        "short_exptime_flag": short_exptime_flag,
        "fit_mask": fit_mask,
        "slope": slope,
        "intercept": intercept,
        "files": files,
    }


def plot_linearity(results, saturation_dn=SATURATION_DN, save_path=None, save_path2=None, title=None):
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

    exptimes = results["exptimes"]
    aperture_vals = results["aperture_vals"]
    fit_mask = results["fit_mask"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(exptimes, aperture_vals, 'o', label="aperture sum", color='#002676')
    axes[0].plot(exptimes[fit_mask], aperture_vals[fit_mask], 'o', color='#FDB515',
                 label="used in linear fit")
    if results["slope"] is not None:
        t_fit = np.linspace(0, exptimes.max(), 100)
        axes[0].plot(t_fit, results["slope"] * t_fit + results["intercept"],
                     '--', color='gray', label="linear fit")
    else:
        print("Note: no fit line drawn -- results['slope'] is None "
              "(fit_mask had fewer than 2 points). See the WARNING "
              "printed by analyze_linearity.")
    axes[0].set_xlabel("Exposure time (s)")
    axes[0].set_ylabel("Aperture sum, bias-subtracted (DN)")
    if title:
        axes[0].set_title(title)
    else:
        axes[0].set_title("Linearity")
    axes[0].legend()

    axes[1].plot(exptimes, results["peak_vals"], 'o', color='#770747')
    axes[1].axhline(saturation_dn, color='grey', linestyle='--', label="Saturation")
    axes[1].set_xlabel("Exposure time (s)")
    axes[1].set_ylabel("Peak pixel value (DN)")
    axes[1].set_title("Peak pixel vs exposure time")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000)
    if save_path2:
        plt.savefig(save_path2, bbox_inches="tight")
            
    return fig