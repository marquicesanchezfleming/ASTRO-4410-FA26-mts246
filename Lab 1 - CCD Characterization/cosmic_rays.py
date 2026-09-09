import glob
import os
from pathlib import Path
from contextlib import contextmanager

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

CCD_PROFILES = {
    "atik_titan": {
        "pixel_pitch_um": 7.4,
        "exptime_s": 600.0,
        "aliases": ["atik", "titan", "crtest"],
    },
    "andor": {
        "pixel_pitch_um": 13.5,
        "exptime_s": 600.0,
        "aliases": ["andor"],
    },
}

CCD_DISPLAY_NAMES = {
    "andor": "Andor",
    "atik_titan": "Atik Titan",
}

HOT_PIXEL_NSIGMA = 5.0      # threshold for "hot pixel"
COSMIC_RAY_NSIGMA = 5.0     # threshold for "cosmic ray"
BAD_FRAME_FRAC = 0.01       # if >1% of a frame's pixels are flagged, it's not cosmic rays -- treat the whole frame as bad


def resolve_ccd_profile(frame_dir, ccd_type=None):
    if ccd_type is not None:
        if ccd_type not in CCD_PROFILES:
            raise ValueError(
                f"Unknown ccd_type '{ccd_type}'. Options: {list(CCD_PROFILES)}")
        return ccd_type, CCD_PROFILES[ccd_type]

    path_lower = frame_dir.lower()
    matches = [name for name, profile in CCD_PROFILES.items() if any(alias in path_lower for alias in profile["aliases"])]
    if len(matches) == 1:
        name = matches[0]
        return name, CCD_PROFILES[name]
    elif len(matches) == 0:
        raise ValueError(
            f"Could not auto-detect CCD type from path:\n  {frame_dir}\n"
            f"Pass ccd_type explicitly, e.g. ccd_type='atik_titan'. "
        )
    else:
        raise ValueError(
            f"Path matches multiple CCD profiles {matches} — ambiguous.\n"
            f"  {frame_dir}\nPass ccd_type explicitly to disambiguate."
        )

def get_exptime(fits_path, keys=("EXPTIME", "EXPOSURE")):
    header = fits.getheader(fits_path)
    for key in keys:
        if key in header:
            return float(header[key])
    return None


def load_stack(frame_dir, exclude_keywords=("bias",),
                expected_exptime_s=None, exptime_tol_s=5.0):
    if not frame_dir.endswith("/"):
        frame_dir = frame_dir + "/"
    all_files = sorted(glob.glob(frame_dir + "*.fit*"))
    if not all_files:
        raise FileNotFoundError(f"No FITS files found in {frame_dir}")
 
    files = [
        f for f in all_files
        if not any(kw.lower() in f.lower() for kw in exclude_keywords)
    ]
    name_excluded = sorted(set(all_files) - set(files))
    if name_excluded:
        print(f"Excluded {len(name_excluded)} file(s) matching {exclude_keywords}:")
        for f in name_excluded:
            print(f"    {f}")
 
    if expected_exptime_s is not None:
        kept = []
        exptime_excluded = []
        for f in files:
            t = get_exptime(f)
            if t is None:
                print(f"WARNING: no exposure time found ...; excluding it")
                continue
            elif abs(t - expected_exptime_s) > exptime_tol_s:
                exptime_excluded.append((f, t))
            else:
                kept.append(f)
        if exptime_excluded:
            print(f"Excluded {len(exptime_excluded)} file(s) with exposure time "
                  f"!= {expected_exptime_s}s (+/- {exptime_tol_s}s):")
            for f, t in exptime_excluded:
                print(f"    {f}  (EXPTIME = {t}s)")
        files = kept
 
    if not files:
        raise FileNotFoundError(f"All files in {frame_dir} were excluded")
 
    stack = np.stack([fits.getdata(f).astype(float) for f in files]) # type: ignore
    print(f"Loaded {len(files)} frames from {frame_dir}, shape {stack.shape}")
    return stack, files
 
 
def find_hot_pixels(stack, nsigma=HOT_PIXEL_NSIGMA):
    median_map = np.median(stack, axis=0)
    mad = np.median(np.abs(median_map - np.median(median_map)))
    sigma = 1.4826 * mad
    if sigma == 0:
        raise ValueError("MAD-based sigma estimate is zero; cannot identify hot pixels.")
    thresh = np.median(median_map) + nsigma * sigma
    hot_mask = median_map > thresh
    print(f"Median map level: {np.median(median_map):.1f} DN, "
          f"robust sigma: {sigma:.2f} DN, threshold: {thresh:.1f} DN")
    print(f"Hot pixels found: {hot_mask.sum()}")
    return hot_mask, median_map
 
 
def find_cosmic_rays(stack, median_map, hot_mask, nsigma=COSMIC_RAY_NSIGMA,
                      bad_frame_frac=BAD_FRAME_FRAC):
    n_frames, ny, nx = stack.shape
    total_pixels = nx * ny
    residuals = stack - median_map[None, :, :]
 
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = 1.4826 * mad
    thresh = nsigma * sigma
    print(f"Residual robust sigma: {sigma:.2f} DN, cosmic-ray threshold: {thresh:.1f} DN")
 
    candidate_masks = np.zeros_like(stack, dtype=bool)
    events_per_frame = []
    for i in range(n_frames):
        candidate = residuals[i] > thresh
        candidate &= ~hot_mask
        candidate_masks[i] = candidate
        events_per_frame.append(candidate.sum())
 
    events_per_frame = np.array(events_per_frame)
    frac_per_frame = events_per_frame / total_pixels
    bad_frames = np.where(frac_per_frame > bad_frame_frac)[0]
    good_frames = np.where(frac_per_frame <= bad_frame_frac)[0]
 
    print(f"Cosmic ray candidates per frame: {events_per_frame}")
    if len(bad_frames) > 0:
        print(f"\n*** {len(bad_frames)} BAD FRAME(S) DETECTED "
              f"(>{bad_frame_frac*100:.1f}% of pixels flagged -- not real cosmic "
              f"ray statistics, likely light leak / saturation / corruption): ***")
        for i in bad_frames:
            print(f"    frame index {i}: {events_per_frame[i]} pixels "
                  f"({frac_per_frame[i]*100:.1f}% of frame) -- excluded from rate")
        print("    -> Inspect these frames directly (e.g. plt.imshow) before "
              "trusting the rate below.\n")
    print(f"Good frames used for rate: {len(good_frames)} / {n_frames}")
    print(f"Mean events/frame (good frames only): "
          f"{events_per_frame[good_frames].mean():.2f}")
 
    return events_per_frame, good_frames, bad_frames, candidate_masks
 
 
def compute_rate(events_per_frame, good_frames, exptime_s, pixel_pitch_um, nx, ny):
    good_events = events_per_frame[good_frames]
    total_events = good_events.sum()
    n_good_frames = len(good_frames)
    total_time_min = n_good_frames * exptime_s / 60.0
 
    pitch_mm = pixel_pitch_um * 1e-3
    area_mm2 = (nx * pitch_mm) * (ny * pitch_mm)
 
    rate = total_events / (total_time_min * area_mm2)
    print(f"\nTotal cosmic ray events (good frames only): {total_events}")
    print(f"Total exposure time (good frames only): {total_time_min:.1f} min")
    print(f"Chip area: {area_mm2:.2f} mm^2")
    print(f"Cosmic ray rate: {rate:.4f} events/min/mm^2")
    return rate
 
 
def analyze(frame_dir, ccd_type=None, exptime_s=None, pixel_pitch_um=None,
            hot_nsigma=HOT_PIXEL_NSIGMA, cr_nsigma=COSMIC_RAY_NSIGMA,
            bad_frame_frac=BAD_FRAME_FRAC, exclude_keywords=("bias",),
            exptime_tol_s=5.0):
    profile_name, profile = resolve_ccd_profile(frame_dir, ccd_type)
    exptime_s = profile["exptime_s"] if exptime_s is None else exptime_s
    pixel_pitch_um = profile["pixel_pitch_um"] if pixel_pitch_um is None else pixel_pitch_um
 
    print(f"CCD profile: {profile_name}  "
          f"(pixel pitch = {pixel_pitch_um} um, exptime = {exptime_s} s)\n")
 
    stack, files = load_stack(frame_dir, exclude_keywords=exclude_keywords,
                               expected_exptime_s=exptime_s, exptime_tol_s=exptime_tol_s)
    ny, nx = stack.shape[1], stack.shape[2]
 
    hot_mask, median_map = find_hot_pixels(stack, nsigma=hot_nsigma)
    events_per_frame, good_frames, bad_frames, candidate_masks = find_cosmic_rays(
        stack, median_map, hot_mask, nsigma=cr_nsigma, bad_frame_frac=bad_frame_frac
    )
 
    if len(bad_frames) > 0:
        print("Bad frame file paths:")
        for i in bad_frames:
            print(f"    [{i}] {files[i]}")
        print()
 
    rate = compute_rate(events_per_frame, good_frames, exptime_s, pixel_pitch_um, nx, ny)
 
    return {
        "ccd_type": profile_name,
        "pixel_pitch_um": pixel_pitch_um,
        "exptime_s": exptime_s,
        "stack": stack,
        "files": files,
        "stack_shape": stack.shape,
        "median_map": median_map,
        "hot_mask": hot_mask,
        "events_per_frame": events_per_frame,
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "bad_frame_paths": [files[i] for i in bad_frames],
        "candidate_masks": candidate_masks,
        "rate_events_per_min_per_mm2": rate,
    }

@contextmanager
def lab_plot_style():
    style = {
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
    }
    with plt.style.context("seaborn-v0_8-white"), plt.rc_context(style):
        yield


def plot_ccd_diagnostics(name, results, output_dir, dpi=300, display_name=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if display_name is None:
        display_name = CCD_DISPLAY_NAMES.get(results.get("ccd_type"), name)

    stack = results["stack"]
    files = results["files"]
    median_map = results["median_map"]
    hot_mask = results["hot_mask"]
    candidate_masks = results["candidate_masks"]

    if len(results["bad_frames"]) == 0 or len(results["good_frames"]) == 0:
        print(f"[{name}] skipping diagnostics: need at least one good and one bad frame")
        return

    good_idx = results["good_frames"][0]
    bad_idx = results["bad_frames"][0]

    with lab_plot_style():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(display_name)
        axes[0].imshow(stack[good_idx],
                        vmin=np.percentile(stack[good_idx], 1),
                        vmax=np.percentile(stack[good_idx], 99), cmap="gray")
        axes[0].set_title(f"Good frame: {os.path.basename(files[good_idx])}")
        axes[1].imshow(stack[bad_idx], cmap="gray")
        axes[1].set_title(f"Bad frame: {os.path.basename(files[bad_idx])}")
        plt.tight_layout()
        fig.savefig(output_dir / f"{name}_good_bad.png", dpi=dpi)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 8))
        plt.imshow(median_map,
                   vmin=np.percentile(median_map, 1),
                   vmax=np.percentile(median_map, 99), cmap="gray")
        ys, xs = np.where(hot_mask)
        plt.scatter(xs, ys, s=10, facecolors="none", edgecolors="red", label="hot pixels")
        plt.legend()
        plt.title(f"{display_name}: median dark map with hot pixels circled")
        fig.savefig(output_dir / f"{name}_median_map.png", dpi=dpi)
        plt.close(fig)

        ys, xs = np.where(candidate_masks[good_idx])
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(stack[good_idx],
                   vmin=np.percentile(stack[good_idx], 1),
                   vmax=np.percentile(stack[good_idx], 99), cmap="gray")
        plt.scatter(xs, ys, s=30, facecolors="none", edgecolors="yellow")
        plt.title(f"{display_name}: cosmic ray candidates, frame {good_idx}")
        fig.savefig(output_dir / f"{name}_cosmic_rays.png", dpi=dpi)
        plt.close(fig)


def full_analyze(andor_dir, atik_dir, output_dir):
    andor_results = analyze(andor_dir)
    atik_results = analyze(atik_dir)

    plot_ccd_diagnostics("andor", andor_results, output_dir)
    plot_ccd_diagnostics("atik", atik_results, output_dir)

    return andor_results, atik_results
