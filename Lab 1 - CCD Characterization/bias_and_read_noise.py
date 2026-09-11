import glob
import numpy as np
from astropy.io import fits
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

def compute_bias_and_read_noise(directory):
    files = sorted(glob.glob(directory + "/*.fit"))

    means = []
    stds = []
    for f in files:
        data = fits.getdata(f).astype(float)
        means.append(np.mean(data))
        stds.append(np.std(data))

    means = np.array(means)
    stds = np.array(stds)

    print("Bias level (mean of means):", np.mean(means))
    print("Read noise (mean of stds):", np.mean(stds))
    print("Std of means (frame-to-frame bias stability):", np.std(means))
    print("Std of stds (frame-to-frame noise consistency):", np.std(stds))

    frame_idx = np.arange(len(files))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(frame_idx, means, 'o', color='#00553A')
    axes[0].axhline(np.mean(means), color='gray', linestyle='--', label=f"mean = {np.mean(means):.2f} DN")
    axes[0].set_xlabel("Frame number")
    axes[0].set_ylabel("Bias level (DN)")
    axes[0].set_title("Bias level per frame")
    axes[0].legend()

    axes[1].plot(frame_idx, stds, 'o', color="#002676")
    axes[1].axhline(np.mean(stds), color='gray', linestyle='--', label=f"mean = {np.mean(stds):.3f} DN")
    axes[1].set_xlabel("Frame number")
    axes[1].set_ylabel("Read noise (DN)")
    axes[1].set_title("Read noise per frame")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 1 - CCD Characterization/plots/bias_readnoise_consistency.png", dpi=1000)
    plt.savefig("/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 1 - CCD Characterization/nicer_plots/bias_readnoise_consistency.pdf", bbox_inches="tight")
    plt.show()