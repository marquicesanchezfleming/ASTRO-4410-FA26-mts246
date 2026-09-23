"""
Generate synthetic optical photometry data for a star cluster.

Produces a CSV of B and V band magnitudes (with realistic photometric
uncertainties) for a population of stars that traces out a main sequence,
a red giant branch, and a handful of white dwarfs -- enough structure to
build a plausible HR diagram (or color-magnitude diagram) from the output.

Usage:
    python generate_fake_star_data.py
    python generate_fake_star_data.py --n-stars 500 --distance-pc 250 --seed 42
"""

import argparse

import numpy as np
import pandas as pd


def teff_to_bv(teff):
    """Approximate blackbody-based Teff (K) -> B-V color conversion."""
    return 0.92 * (5040.0 / teff) - 0.7


def main_sequence_abs_mag(bv):
    """
    Rough empirical fit of absolute V magnitude vs. B-V color for
    main-sequence stars (calibrated loosely to Sun at B-V=0.65, M_V=4.83).
    """
    return 5.6 + 8.0 * bv - 3.0 * bv**2 + 4.0 * bv**3


def generate_main_sequence(n_stars, rng):
    teff = rng.uniform(3500, 20000, n_stars)
    bv = teff_to_bv(teff)
    m_v = main_sequence_abs_mag(bv) + rng.normal(0, 0.35, n_stars)
    return bv, m_v


def generate_giant_branch(n_stars, rng):
    bv = rng.uniform(0.8, 1.8, n_stars)
    # Giants sit well above the main sequence at a given color.
    m_v = rng.uniform(-1.0, 1.5, n_stars) - 2.0 * (bv - 0.8)
    return bv, m_v


def generate_white_dwarfs(n_stars, rng):
    bv = rng.uniform(-0.3, 0.6, n_stars)
    m_v = rng.uniform(10.5, 13.5, n_stars)
    return bv, m_v


def generate_cluster(n_stars, giant_fraction, wd_fraction, rng):
    n_giants = int(round(n_stars * giant_fraction))
    n_wd = int(round(n_stars * wd_fraction))
    n_ms = n_stars - n_giants - n_wd

    bv_ms, mv_ms = generate_main_sequence(n_ms, rng)
    bv_gb, mv_gb = generate_giant_branch(n_giants, rng)
    bv_wd, mv_wd = generate_white_dwarfs(n_wd, rng)

    bv = np.concatenate([bv_ms, bv_gb, bv_wd])
    m_v = np.concatenate([mv_ms, mv_gb, mv_wd])
    evol_stage = np.array(
        ["main_sequence"] * n_ms + ["giant"] * n_giants + ["white_dwarf"] * n_wd
    )
    return bv, m_v, evol_stage


def add_photometric_errors(v_app, b_app, rng, base_err=0.01, faint_scale=0.03):
    """Fainter stars get noisier magnitudes, mimicking real photometry."""
    v_err = base_err + faint_scale * 10 ** ((v_app - 15) / 5)
    b_err = base_err + faint_scale * 10 ** ((b_app - 15) / 5)
    v_err = np.clip(v_err, 0.005, 0.5)
    b_err = np.clip(b_err, 0.005, 0.5)

    v_obs = v_app + rng.normal(0, v_err)
    b_obs = b_app + rng.normal(0, b_err)
    return v_obs, v_err, b_obs, b_err


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-stars", type=int, default=300, help="Number of stars to simulate")
    parser.add_argument("--distance-pc", type=float, default=150.0, help="Cluster distance in parsecs")
    parser.add_argument("--giant-fraction", type=float, default=0.08, help="Fraction of stars on the giant branch")
    parser.add_argument("--wd-fraction", type=float, default=0.04, help="Fraction of stars that are white dwarfs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=str,
        default="fake_star_data.csv",
        help="Output CSV filename (written next to this script)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    bv_true, m_v_true, evol_stage = generate_cluster(
        args.n_stars, args.giant_fraction, args.wd_fraction, rng
    )

    distance_modulus = 5 * np.log10(args.distance_pc / 10)
    v_app_true = m_v_true + distance_modulus
    b_app_true = v_app_true + bv_true

    v_obs, v_err, b_obs, b_err = add_photometric_errors(v_app_true, b_app_true, rng)

    star_id = np.array([f"star_{i:04d}" for i in range(len(v_obs))])

    df = pd.DataFrame(
        {
            "star_id": star_id,
            "B_mag": np.round(b_obs, 3),
            "B_err": np.round(b_err, 3),
            "V_mag": np.round(v_obs, 3),
            "V_err": np.round(v_err, 3),
            "evolutionary_stage": evol_stage,
        }
    )
    # Shuffle rows so evolutionary stage isn't trivially sorted in the file.
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    output_path = args.output
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} stars to {output_path}")
    print(f"Assumed cluster distance: {args.distance_pc} pc (distance modulus = {distance_modulus:.2f})")


if __name__ == "__main__":
    main()
