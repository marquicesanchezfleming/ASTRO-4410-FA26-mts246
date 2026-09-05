import glob
import numpy as np
from astropy.io import fits

FRAME_DIR = "/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 1 - CCD Characterization/CRtest"      
EXPTIME_S = 600.0                    # 10 minutes in seconds
PIXEL_PITCH_UM = 7.4                 # 7.4 for Atik Titan, 13.5 for Andor

HOT_PIXEL_NSIGMA = 8.0               # median map outlier threshold for "hot pixel"
COSMIC_RAY_NSIGMA = 5.0              # per-frame residual outlier threshold for "cosmic ray"

def load_stack(frame_dir):
    files = sorted(glob.glob(frame_dir + "*.fit*"))
    if not files:
        raise FileNotFoundError(f"No FITS files found in {frame_dir}")
    stack = np.stack([fits.getdata(f).astype(float) for f in files])
    print(f"Loaded {len(files)} frames, shape {stack.shape}")
    return stack, files


def find_hot_pixels(stack, nsigma=HOT_PIXEL_NSIGMA):
    median_map = np.median(stack, axis=0)
    mad = np.median(np.abs(median_map - np.median(median_map)))
    sigma = 1.4826 * mad
    thresh = np.median(median_map) + nsigma * sigma
    hot_mask = median_map > thresh
    print(f"Median map level: {np.median(median_map):.1f} DN, "
          f"robust sigma: {sigma:.2f} DN, threshold: {thresh:.1f} DN")
    print(f"Hot pixels found: {hot_mask.sum()}")
    return hot_mask, median_map


def find_cosmic_rays(stack, median_map, hot_mask, nsigma=COSMIC_RAY_NSIGMA):
    n_frames = stack.shape[0]
    residuals = stack - median_map[None, :, :]

    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = 1.4826 * mad
    thresh = nsigma * sigma
    print(f"Residual robust sigma: {sigma:.2f} DN, cosmic-ray threshold: {thresh:.1f} DN")

    events_per_frame = []
    for i in range(n_frames):
        candidate = residuals[i] > thresh
        candidate &= ~hot_mask  
        n_events = candidate.sum()
        events_per_frame.append(n_events)

    events_per_frame = np.array(events_per_frame)
    print(f"Cosmic ray candidates per frame: {events_per_frame}")
    print(f"Mean events/frame: {events_per_frame.mean():.2f}")
    return events_per_frame


def compute_rate(events_per_frame, exptime_s, pixel_pitch_um, nx, ny):
    total_events = events_per_frame.sum()
    n_frames = len(events_per_frame)
    total_time_min = n_frames * exptime_s / 60.0

    pitch_mm = pixel_pitch_um * 1e-3
    area_mm2 = nx * pitch_mm * ny * pitch_mm

    rate = total_events / (total_time_min * area_mm2)
    print(f"\nTotal cosmic ray events: {total_events}")
    print(f"Total exposure time: {total_time_min:.1f} min")
    print(f"Chip area: {area_mm2:.2f} mm^2")
    print(f"Cosmic ray rate: {rate:.4f} events/min/mm^2")
    return rate