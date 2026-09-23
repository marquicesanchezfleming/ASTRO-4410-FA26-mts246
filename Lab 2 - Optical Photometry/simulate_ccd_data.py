"""
Simulate raw CCD frames of a star cluster field, the way they'd actually
come off a telescope -- uncalibrated FITS images full of ADU counts, not a
ready-made table of magnitudes like fake_star_data.csv.

A real optical-photometry run produces four kinds of frames:
  - bias frames  : 0 s exposures            -> electronic offset + read noise
  - dark frames  : shutter closed, exptime matched to the science frames
                   -> thermal (dark-current) signal + bias + read noise
  - flat frames  : uniformly illuminated (twilight/dome), one set per filter
                   -> pixel-to-pixel and large-scale sensitivity variations
  - light frames : the actual science exposures of the cluster, per filter,
                   several dithered exposures per filter

None of these are magnitudes yet. Every frame is a 2D array of integer ADU
plus a FITS header (EXPTIME, FILTER, GAIN, RDNOISE, IMAGETYP, a WCS, ...).
Getting from these to a B/V magnitude per star -- master bias/dark/flat,
calibrating the light frames, aperture photometry, then an instrumental-to
-standard magnitude transform -- is the actual lab; that part is left for
the notebook.

This script forward-models the same star-cluster population used in
generate_fake_star_data.py (a main sequence + giant branch + a few white
dwarfs, at a chosen distance) through a simple telescope + CCD:
    apparent magnitude
        -> electrons/s   (photon zero point, telescope area, QE, extinction)
        -> electrons collected (Poisson shot noise, Gaussian PSF, sky
           background, dark current, hot/dead pixels)
        -> ADU            (flat-field response, read noise, bias, gain,
           saturation, 16-bit digitization)

Usage:
    python simulate_ccd_data.py
    python simulate_ccd_data.py --n-stars 300 --distance-pc 300 --seed 7
"""

import argparse
import os

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.special import erf

from generate_fake_star_data import generate_cluster

# ---------------------------------------------------------------------------
# Ground truth: CCD, telescope, and site. Change these and every frame you
# generate (bias/dark/flat/light) will reflect it consistently, same as a
# real chip -- this is the "TRUE_CCD" your reduction pipeline should recover.
# ---------------------------------------------------------------------------

CCD = {
    "shape": (512, 512),            # (ny, nx) pixels
    "pixel_scale_arcsec": 0.50,     # arcsec / pixel (plate scale)
    "gain_e_per_adu": 1.60,         # electrons per ADU
    "read_noise_e": 7.5,            # electrons, per pixel per read
    "bias_level_adu": 500.0,        # ADU
    "dark_current_e_per_s": 0.04,   # electrons / s / pixel (cooled CCD)
    "full_well_e": 100_000.0,       # electron full-well capacity
    "adc_bits": 16,
    "qe": 0.80,                     # quantum efficiency
    "fixed_pattern_frac": 0.015,    # pixel-to-pixel flat response scatter (1 sigma)
    "n_hot_pixels": 12,
    "hot_pixel_rate_e_per_s": 400.0,
    "n_dead_pixels": 6,
    "n_dust_donuts": 3,
    "flat_target_rate_e_s": 9000.0,  # illumination level used for flat exposures
}

TELESCOPE = {
    "aperture_diameter_m": 0.60,
    "obstruction_frac": 0.35,   # fractional area blocked by secondary mirror
    "throughput": 0.65,         # optics + filter transmission
}

SITE = {
    "airmass": 1.2,
    "seeing_fwhm_arcsec": 2.2,
    "extinction_mag_per_airmass": {"B": 0.28, "V": 0.16},
    "sky_mag_per_arcsec2": {"B": 21.5, "V": 20.7},
}

# Johnson B/V photon zero points, built from the standard flux-calibration
# constants (Bessell 1998): F_lambda at m=0, effective wavelength, and FWHM
# bandwidth. Converting to a photon *rate* (rather than energy flux) is what
# actually lets you count electrons for a given exposure time.
FILTER_CALIBRATION = {
    "B": {"f_lambda0_erg_s_cm2_A": 6.32e-9, "lambda_eff_A": 4400.0, "bandwidth_A": 980.0},
    "V": {"f_lambda0_erg_s_cm2_A": 3.63e-9, "lambda_eff_A": 5500.0, "bandwidth_A": 890.0},
}

H_ERG_S = 6.626e-27
C_A_S = 2.998e18  # speed of light, Angstrom/s

# A fictitious field -- not real survey data, just a plausible RA/Dec to
# hang a WCS on.
FIELD_CENTER_RA_DEG = 132.849
FIELD_CENTER_DEC_DEG = 11.800


def photon_zeropoint_rate(filt_name):
    """Photon flux (photons / s / cm^2) an m=0 star delivers, integrated over the bandpass."""
    cal = FILTER_CALIBRATION[filt_name]
    e_photon_erg = H_ERG_S * C_A_S / cal["lambda_eff_A"]
    return cal["f_lambda0_erg_s_cm2_A"] * cal["bandwidth_A"] / e_photon_erg


ZERO_POINT_PHOTON_RATE = {f: photon_zeropoint_rate(f) for f in FILTER_CALIBRATION}


def collecting_area_cm2(telescope):
    d_cm = telescope["aperture_diameter_m"] * 100.0
    return np.pi / 4.0 * d_cm**2 * (1.0 - telescope["obstruction_frac"])


def star_electron_rate(m_app, filt_name, telescope, ccd, site):
    """Electrons/s a star of apparent magnitude m_app delivers to the detector."""
    k = site["extinction_mag_per_airmass"][filt_name]
    m_observed = m_app + k * site["airmass"]
    photon_rate = (
        ZERO_POINT_PHOTON_RATE[filt_name]
        * 10 ** (-0.4 * m_observed)
        * collecting_area_cm2(telescope)
        * telescope["throughput"]
    )
    return photon_rate * ccd["qe"]


def sky_electron_rate_per_pixel(filt_name, site, telescope, ccd):
    """Electrons/s/pixel from uniform sky background (already an observed, extincted quantity)."""
    mu = site["sky_mag_per_arcsec2"][filt_name]
    pixel_area_arcsec2 = ccd["pixel_scale_arcsec"] ** 2
    m_pixel_equiv = mu - 2.5 * np.log10(pixel_area_arcsec2)
    photon_rate = (
        ZERO_POINT_PHOTON_RATE[filt_name]
        * 10 ** (-0.4 * m_pixel_equiv)
        * collecting_area_cm2(telescope)
        * telescope["throughput"]
    )
    return photon_rate * ccd["qe"]


# ---------------------------------------------------------------------------
# Spatial layout: a Plummer-like (King-ish) radial concentration toward the
# field center, so the cluster actually looks like a cluster instead of a
# scatter plot.
# ---------------------------------------------------------------------------

def place_stars_on_chip(n_stars, shape, core_radius_px, rng, margin_px=15):
    ny, nx = shape
    cx, cy = nx / 2.0, ny / 2.0
    max_r = min(cx, cy) - margin_px

    unif = rng.uniform(0, 0.97, n_stars)  # cap so r stays finite/bounded
    r = core_radius_px * np.sqrt(unif / (1 - unif))
    r = np.clip(r, 0, max_r)
    theta = rng.uniform(0, 2 * np.pi, n_stars)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return x, y


# ---------------------------------------------------------------------------
# Pixel-level rendering
# ---------------------------------------------------------------------------

def render_star_stamp(shape, x0, y0, total_electrons, sigma_px):
    """Flux-conserving Gaussian PSF, exactly integrated over each pixel via erf."""
    ny, nx = shape
    half = int(np.ceil(6 * sigma_px)) + 1
    xlo, xhi = max(0, int(x0) - half), min(nx, int(x0) + half + 1)
    ylo, yhi = max(0, int(y0) - half), min(ny, int(y0) + half + 1)
    if xhi <= xlo or yhi <= ylo:
        return None

    sqrt2 = np.sqrt(2)
    x_idx = np.arange(xlo, xhi)
    y_idx = np.arange(ylo, yhi)
    fx = 0.5 * (erf((x_idx + 0.5 - x0) / (sqrt2 * sigma_px)) - erf((x_idx - 0.5 - x0) / (sqrt2 * sigma_px)))
    fy = 0.5 * (erf((y_idx + 0.5 - y0) / (sqrt2 * sigma_px)) - erf((y_idx - 0.5 - y0) / (sqrt2 * sigma_px)))
    stamp = total_electrons * np.outer(fy, fx)
    return ylo, yhi, xlo, xhi, stamp


def make_flat_field_pattern(shape, ccd, rng):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny / 2.0, nx / 2.0
    r_norm = np.hypot(xx - cx, yy - cy) / np.hypot(cx, cy)
    vignette = 1.0 - 0.12 * r_norm**2
    pixel_response = 1.0 + rng.normal(0, ccd["fixed_pattern_frac"], size=shape)
    pattern = vignette * pixel_response

    for _ in range(ccd["n_dust_donuts"]):
        dx, dy = rng.uniform(0.15, 0.85, 2) * [nx, ny]
        r_donut = rng.uniform(20, 45)
        depth = rng.uniform(0.03, 0.08)
        rr = np.hypot(xx - dx, yy - dy)
        pattern *= 1.0 - depth * np.exp(-((rr - r_donut) ** 2) / (2 * (r_donut * 0.25) ** 2))
    return pattern


def make_defect_maps(shape, ccd, rng):
    ny, nx = shape
    hot_y = rng.integers(0, ny, ccd["n_hot_pixels"])
    hot_x = rng.integers(0, nx, ccd["n_hot_pixels"])
    dead_y = rng.integers(0, ny, ccd["n_dead_pixels"])
    dead_x = rng.integers(0, nx, ccd["n_dead_pixels"])
    return hot_y, hot_x, dead_y, dead_x


def build_frame(shape, exptime, ccd, rng, flat_pattern, defects,
                 frame_kind="light", sky_rate=0.0, stars=None, sigma_px=None):
    """
    frame_kind: "bias" | "dark" | "flat" | "light"
    stars: dict with arrays x_px, y_px, electron_rate (e-/s), only used for "light"
    """
    hot_y, hot_x, dead_y, dead_x = defects
    mean_e = np.zeros(shape, dtype=np.float64)

    if frame_kind == "light":
        mean_e += sky_rate * exptime
        if stars is not None:
            for x0, y0, rate in zip(stars["x_px"], stars["y_px"], stars["electron_rate"]):
                stamp = render_star_stamp(shape, x0, y0, rate * exptime, sigma_px)
                if stamp is not None:
                    ylo, yhi, xlo, xhi, arr = stamp
                    mean_e[ylo:yhi, xlo:xhi] += arr
        mean_e *= flat_pattern
    elif frame_kind == "flat":
        mean_e += ccd["flat_target_rate_e_s"] * exptime * flat_pattern
    # "bias" / "dark": no light term

    mean_e += ccd["dark_current_e_per_s"] * exptime
    if len(hot_y):
        mean_e[hot_y, hot_x] += ccd["hot_pixel_rate_e_per_s"] * exptime

    mean_e = np.clip(mean_e, 0, None)
    e_signal = rng.poisson(mean_e).astype(np.float64)
    e_total = e_signal + rng.normal(0, ccd["read_noise_e"], size=shape)

    if len(dead_y):
        e_total[dead_y, dead_x] = rng.normal(0, ccd["read_noise_e"], size=len(dead_y))

    e_total = np.minimum(e_total, ccd["full_well_e"])
    adu = ccd["bias_level_adu"] + e_total / ccd["gain_e_per_adu"]
    adu = np.clip(adu, 0, 2 ** ccd["adc_bits"] - 1)
    return np.round(adu).astype(np.uint16)


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------

def make_header(exptime, filt_name, frame_kind, ccd, site, dither_px=(0.0, 0.0), obs_time=None):
    ny, nx = ccd["shape"]
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2.0 + dither_px[0], ny / 2.0 + dither_px[1]]
    w.wcs.cdelt = [-ccd["pixel_scale_arcsec"] / 3600.0, ccd["pixel_scale_arcsec"] / 3600.0]
    w.wcs.crval = [FIELD_CENTER_RA_DEG, FIELD_CENTER_DEC_DEG]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    header = w.to_header()
    header["SIMPLE"] = True
    header["IMAGETYP"] = {"bias": "BIAS", "dark": "DARK", "flat": "FLAT", "light": "LIGHT"}[frame_kind]
    header["OBJECT"] = "SIM-CLUSTER" if frame_kind == "light" else frame_kind.upper()
    header["EXPTIME"] = (float(exptime), "seconds")
    header["FILTER"] = filt_name if filt_name else "NONE"
    header["GAIN"] = (ccd["gain_e_per_adu"], "electrons/ADU")
    header["RDNOISE"] = (ccd["read_noise_e"], "electrons")
    header["CCD-TEMP"] = (-20.0, "deg C")
    header["AIRMASS"] = (site["airmass"], None)
    header["TELESCOP"] = "SIM 0.6m"
    header["INSTRUME"] = "SimCCD-512"
    header["PIXSCALE"] = (ccd["pixel_scale_arcsec"], "arcsec/pixel")
    if obs_time is not None:
        header["DATE-OBS"] = obs_time.isot
    return header


def write_fits(path, data, header):
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all(
    output_dir,
    n_stars=300,
    distance_pc=1200.0,
    giant_fraction=0.08,
    wd_fraction=0.04,
    core_radius_arcsec=40.0,
    exptime_light=60.0,
    exptime_flat=3.0,
    n_bias=9,
    n_dark=5,
    n_flat=5,
    n_light=5,
    seed=None,
):
    """
    Generate one full raw-data set (bias/dark/flat/light FITS frames + a
    ground-truth catalog) under output_dir. This is the notebook-callable
    entry point -- everything main()/the CLI does is just argument parsing
    around this one call. Returns the ground-truth catalog as a DataFrame.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    ccd, telescope, site = CCD, TELESCOPE, SITE
    shape = ccd["shape"]

    os.makedirs(f"{output_dir}/bias", exist_ok=True)
    os.makedirs(f"{output_dir}/dark", exist_ok=True)
    os.makedirs(f"{output_dir}/flat", exist_ok=True)
    os.makedirs(f"{output_dir}/light", exist_ok=True)

    # --- fixed detector properties, shared by every frame this run produces ---
    flat_pattern = make_flat_field_pattern(shape, ccd, rng)
    defects = make_defect_maps(shape, ccd, rng)

    # --- cluster population (reuses the same physics as generate_fake_star_data.py) ---
    bv_true, m_v_true, evol_stage = generate_cluster(n_stars, giant_fraction, wd_fraction, rng)
    distance_modulus = 5 * np.log10(distance_pc / 10)
    v_app = m_v_true + distance_modulus
    b_app = v_app + bv_true

    core_radius_px = core_radius_arcsec / ccd["pixel_scale_arcsec"]
    x_px, y_px = place_stars_on_chip(n_stars, shape, core_radius_px, rng)

    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2.0, shape[0] / 2.0]
    w.wcs.cdelt = [-ccd["pixel_scale_arcsec"] / 3600.0, ccd["pixel_scale_arcsec"] / 3600.0]
    w.wcs.crval = [FIELD_CENTER_RA_DEG, FIELD_CENTER_DEC_DEG]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ra_deg, dec_deg = w.wcs_pix2world(x_px, y_px, 0)

    sigma_px = (site["seeing_fwhm_arcsec"] / ccd["pixel_scale_arcsec"]) / 2.3548

    obs_time = Time("2026-01-15T05:00:00")

    # --- bias frames ---
    for i in range(n_bias):
        img = build_frame(shape, 0.0, ccd, rng, flat_pattern, defects, frame_kind="bias")
        hdr = make_header(0.0, None, "bias", ccd, site, obs_time=obs_time)
        write_fits(f"{output_dir}/bias/bias_{i:03d}.fits", img, hdr)
        obs_time += 30 * u.s

    # --- dark frames (matched to the science exposure time) ---
    for i in range(n_dark):
        img = build_frame(shape, exptime_light, ccd, rng, flat_pattern, defects, frame_kind="dark")
        hdr = make_header(exptime_light, None, "dark", ccd, site, obs_time=obs_time)
        write_fits(f"{output_dir}/dark/dark_{exptime_light:.0f}s_{i:03d}.fits", img, hdr)
        obs_time += (exptime_light + 10) * u.s

    # --- flat frames, per filter ---
    for filt in ("B", "V"):
        for i in range(n_flat):
            img = build_frame(shape, exptime_flat, ccd, rng, flat_pattern, defects, frame_kind="flat")
            hdr = make_header(exptime_flat, filt, "flat", ccd, site, obs_time=obs_time)
            write_fits(f"{output_dir}/flat/flat_{filt}_{i:03d}.fits", img, hdr)
            obs_time += (exptime_flat + 5) * u.s

    # --- light frames, per filter, dithered ---
    # Fraction of a star's total flux that lands in its single brightest pixel
    # (worst case: centered exactly on that pixel), from the same erf pixel
    # integration used to render the PSF -- needed to flag saturation from a
    # *peak-pixel* electron count, not the star's total integrated electrons.
    peak_pixel_frac = (erf(0.5 / (np.sqrt(2) * sigma_px))) ** 2

    saturated_any = np.zeros(n_stars, dtype=bool)
    for filt in ("B", "V"):
        m_app = b_app if filt == "B" else v_app
        electron_rate = star_electron_rate(m_app, filt, telescope, ccd, site)
        peak_e = electron_rate * exptime_light * peak_pixel_frac
        saturated_any |= peak_e > ccd["full_well_e"]

        for i in range(n_light):
            dx, dy = rng.uniform(-3, 3, 2)
            stars = {"x_px": x_px + dx, "y_px": y_px + dy, "electron_rate": electron_rate}
            sky_rate = sky_electron_rate_per_pixel(filt, site, telescope, ccd)
            img = build_frame(
                shape, exptime_light, ccd, rng, flat_pattern, defects,
                frame_kind="light", sky_rate=sky_rate, stars=stars, sigma_px=sigma_px,
            )
            hdr = make_header(exptime_light, filt, "light", ccd, site, dither_px=(dx, dy), obs_time=obs_time)
            write_fits(f"{output_dir}/light/light_{filt}_{i:03d}.fits", img, hdr)
            obs_time += (exptime_light + 15) * u.s

    # --- ground-truth catalog, for checking your reduction against once you're done ---
    star_id = np.array([f"star_{i:04d}" for i in range(n_stars)])
    truth = pd.DataFrame(
        {
            "star_id": star_id,
            "x_px": np.round(x_px, 2),
            "y_px": np.round(y_px, 2),
            "ra_deg": np.round(ra_deg, 6),
            "dec_deg": np.round(dec_deg, 6),
            "B_true_app": np.round(b_app, 3),
            "V_true_app": np.round(v_app, 3),
            "evolutionary_stage": evol_stage,
            "saturates_in_light_frame": saturated_any,
        }
    )
    truth_path = f"{output_dir}/truth_catalog_ANSWER_KEY.csv"
    truth.to_csv(truth_path, index=False)

    print(f"Wrote {n_bias} bias, {n_dark} dark, "
          f"{2 * n_flat} flat, {2 * n_light} light frames to {output_dir}/")
    print(f"Cluster: {n_stars} stars at {distance_pc} pc "
          f"(distance modulus {distance_modulus:.2f}), field center "
          f"RA={FIELD_CENTER_RA_DEG:.4f} Dec={FIELD_CENTER_DEC_DEG:.4f}")
    print(f"{saturated_any.sum()} star(s) saturate in a {exptime_light:.0f}s light frame "
          f"-- expected for the brightest cluster members, exclude them from photometry.")
    print(f"Ground-truth catalog (for checking your results, not something you'd have for "
          f"real data): {truth_path}")

    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-stars", type=int, default=300)
    parser.add_argument("--distance-pc", type=float, default=1200.0)
    parser.add_argument("--giant-fraction", type=float, default=0.08)
    parser.add_argument("--wd-fraction", type=float, default=0.04)
    parser.add_argument("--core-radius-arcsec", type=float, default=40.0)
    parser.add_argument("--exptime-light", type=float, default=60.0)
    parser.add_argument("--exptime-flat", type=float, default=3.0)
    parser.add_argument("--n-bias", type=int, default=9)
    parser.add_argument("--n-dark", type=int, default=5)
    parser.add_argument("--n-flat", type=int, default=5, help="per filter")
    parser.add_argument("--n-light", type=int, default=5, help="per filter")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="raw_data")
    args = parser.parse_args()

    generate_all(
        args.output_dir,
        n_stars=args.n_stars,
        distance_pc=args.distance_pc,
        giant_fraction=args.giant_fraction,
        wd_fraction=args.wd_fraction,
        core_radius_arcsec=args.core_radius_arcsec,
        exptime_light=args.exptime_light,
        exptime_flat=args.exptime_flat,
        n_bias=args.n_bias,
        n_dark=args.n_dark,
        n_flat=args.n_flat,
        n_light=args.n_light,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
