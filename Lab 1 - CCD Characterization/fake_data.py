"""
NOTE: This script, as well as the subsequent synthetic data, is AI-generated. 
It is intended to provide a consistent set of test data for the CCD 
characterization lab, but it may not reflect real-world CCD behavior 
accurately. Use it for educational purposes only.

One consistent synthetic CCD, used to generate test data for every
section of Lab 1 (A/C/E bias & darks, B linearity, D photon transfer).

Unlike the earlier one-off generators (each of which invented its own
independent ground truth just to test one script), everything here
draws from a single TRUE_CCD dict. That means the numbers you recover
from each section's analysis script should agree with each other and
with TRUE_CCD -- which is exactly the kind of cross-section
consistency your actual report will want to demonstrate.
 
"""

import os
import numpy as np
from astropy.io import fits

TRUE_CCD = {
    "shape": (200, 200),
    "bias_level": 1000.0,          # DN
    "read_noise_dn": 8.0,          # DN
    "gain": 2.5,                   # DN per electron
    "dark_rate_cooler_off_dn_s": 8.0,
    "dark_rate_cooler_on_dn_s": 0.5,
    "n_hot_pixels": 15,
    "hot_pixel_rate_dn_s": 200.0,
    "fixed_pattern_frac": 0.02,    # pixel-to-pixel response variation
    "saturation_dn": 65000.0,
    "star_peak_rate_dn_s": 650.0,  # for the Section B linearity series
    "star_center": (100, 100),
    "star_fwhm_px": 4.0,
}


def _write(path, image, exptime):
    hdu = fits.PrimaryHDU(data=image.astype(np.float32))
    hdu.header["EXPTIME"] = float(exptime)
    hdu.writeto(path, overwrite=True)


def _hot_pixel_locations(params, seed=1):
    rng = np.random.default_rng(seed)
    ny, nx = params["shape"]
    ys = rng.integers(0, ny, params["n_hot_pixels"])
    xs = rng.integers(0, nx, params["n_hot_pixels"])
    return ys, xs


def _fixed_pattern_map(params, seed=2):
    rng = np.random.default_rng(seed)
    return 1.0 + rng.normal(0, params["fixed_pattern_frac"], size=params["shape"])


def generate_bias_frames(output_dir, n_frames=20, params=TRUE_CCD, seed=10):
    """Section A: zero-second exposures -- bias + read noise only."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_frames):
        image = params["bias_level"] + rng.normal(0, params["read_noise_dn"], params["shape"])
        _write(f"{output_dir}/bias_{i:03d}.fits", image, exptime=0.0)
    print(f"Wrote {n_frames} bias frames to {output_dir}")


def generate_dark_series(output_dir, cooler="off", exptimes=(0.1, 1, 5, 10, 20, 40, 80, 160, 320),
                          params=TRUE_CCD, seed=20):
    """Section C: lens-capped exposures at increasing exptime, with
    hot pixels included so the region-median vs single-pixel
    comparison has something real to demonstrate."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    dark_rate = params[f"dark_rate_cooler_{cooler}_dn_s"]
    hot_ys, hot_xs = _hot_pixel_locations(params)

    for i, t in enumerate(exptimes):
        base = params["bias_level"] + dark_rate * t
        image = rng.poisson(np.clip(base, 0, None), size=params["shape"]).astype(float)
        image += rng.normal(0, params["read_noise_dn"], params["shape"])
        image[hot_ys, hot_xs] += params["hot_pixel_rate_dn_s"] * t
        _write(f"{output_dir}/dark_{cooler}_{i:03d}_{t}s.fits", image, exptime=t)
    print(f"Wrote {len(exptimes)} cooler-{cooler} dark frames to {output_dir}")


def generate_linearity_series(output_dir,
                                exptimes=(0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 45, 60, 80, 100),
                                params=TRUE_CCD, seed=30):
    """Section B: point source at increasing exposure time, saturating
    near the top of the range."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    ny, nx = params["shape"]
    yy, xx = np.mgrid[0:ny, 0:nx]
    sigma = params["star_fwhm_px"] / 2.3548
    cx, cy = params["star_center"]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    psf_shape = np.exp(-r2 / (2 * sigma ** 2))

    for i, t in enumerate(exptimes):
        peak = params["star_peak_rate_dn_s"] * t
        star = peak * psf_shape
        image = params["bias_level"] + star
        image = rng.poisson(np.clip(image, 0, None), size=params["shape"]).astype(float)
        image += rng.normal(0, params["read_noise_dn"], params["shape"])
        image = np.clip(image, 0, params["saturation_dn"] + 500)
        _write(f"{output_dir}/linearity_{i:03d}_{t}s.fits", image, exptime=t)
    print(f"Wrote {len(exptimes)} linearity frames to {output_dir}")


def generate_ptc_series(output_dir,
                          target_signals_dn=(5, 15, 50, 150, 400, 1000, 3000, 8000, 20000, 45000),
                          params=TRUE_CCD, seed=40):
    """Section D: diffuse-source flats spanning read-noise-limited to
    fixed-pattern-limited signal levels."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    response_map = _fixed_pattern_map(params)
    gain = params["gain"]

    for i, s_dn in enumerate(target_signals_dn):
        Ne_mean = s_dn / gain
        Ne = rng.poisson(np.clip(Ne_mean * response_map, 0, None)).astype(float)
        image = params["bias_level"] + gain * Ne
        image += rng.normal(0, params["read_noise_dn"], params["shape"])
        _write(f"{output_dir}/flat_{i:03d}.fits", image, exptime=float(i + 1))
    print(f"Wrote {len(target_signals_dn)} PTC flat frames to {output_dir}")


def generate_cosmic_ray_series(output_dir, n_frames=24, exptime=600.0,
                                 cosmic_rays_per_frame=5, n_stray_bias=3,
                                 n_stray_short_dark=1, params=TRUE_CCD, seed=50):
    """
    Section E: long darks with injected cosmic ray hits, PLUS a few
    stray bias/short-dark files mixed in by filename and/or header --
    same contamination pattern you found in your real Andor dataset,
    so you can re-check that the filename + header exptime filtering
    in cosmic_rays.py still catches them correctly.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    ny, nx = params["shape"]
    dark_rate = params["dark_rate_cooler_off_dn_s"]
    hot_ys, hot_xs = _hot_pixel_locations(params)

    for i in range(n_frames):
        base = params["bias_level"] + dark_rate * exptime
        image = rng.poisson(np.clip(base, 0, None), size=params["shape"]).astype(float)
        image += rng.normal(0, params["read_noise_dn"], params["shape"])
        image[hot_ys, hot_xs] += params["hot_pixel_rate_dn_s"] * exptime

        n_hits = rng.poisson(cosmic_rays_per_frame)
        hit_ys = rng.integers(0, ny, n_hits)
        hit_xs = rng.integers(0, nx, n_hits)
        image[hit_ys, hit_xs] += rng.uniform(500, 5000, n_hits)

        _write(f"{output_dir}/Dark-{i:03d}_dark.fits", image, exptime=exptime)

    for i in range(n_stray_bias):
        image = params["bias_level"] + rng.normal(0, params["read_noise_dn"], params["shape"])
        _write(f"{output_dir}/Dark-{i:03d}_bias.fits", image, exptime=0.0)

    for i in range(n_stray_short_dark):
        short_t = 1.0
        base = params["bias_level"] + dark_rate * short_t
        image = rng.poisson(np.clip(base, 0, None), size=params["shape"]).astype(float)
        image += rng.normal(0, params["read_noise_dn"], params["shape"])
        _write(f"{output_dir}/Dark-stray{i:03d}_dark.fits", image, exptime=short_t)

    print(f"Wrote {n_frames} cosmic-ray darks + {n_stray_bias} stray bias + "
          f"{n_stray_short_dark} stray short dark to {output_dir}")


def generate_all(base_dir, params=TRUE_CCD):
    """Generate every section's test data under base_dir, from the
    same TRUE_CCD ground truth."""
    if not base_dir.endswith("/"):
        base_dir += "/"
    generate_bias_frames(base_dir + "section_A_bias/", params=params)
    generate_dark_series(base_dir + "section_C_dark_off/", cooler="off", params=params)
    generate_dark_series(base_dir + "section_C_dark_on/", cooler="on", params=params)
    generate_linearity_series(base_dir + "section_B_linearity/", params=params)
    generate_ptc_series(base_dir + "section_D_ptc/", params=params)
    generate_cosmic_ray_series(base_dir + "section_E_cosmic_rays/", params=params)
    print("\nGround truth used for all of the above:")
    for k, v in params.items():
        print(f"    {k}: {v}")
