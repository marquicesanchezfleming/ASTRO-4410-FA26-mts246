"""
Generate a mock 'data_lime_multispec_...npz' file matching the exact schema
expected by read_Lime_multispec_A4410.py (JMC's Cornell A4410 HI pipeline).

Scenario simulated: a FIXED Az/El pointing (i.e. a real drift scan -- the
telescope is parked and the sky drifts through the beam as the Earth
rotates), observed from Ithaca, NY.

The HI emission line is placed at the true LSR/barycentric Doppler-shifted
frequency using the same calculation the analysis script performs
(directionvec.radial_velocity_correction() + dot with the solar motion
vsun_uvw), so the dashed "expected line center vs. time" overlay drawn by
plot_ds() in the analysis script will line up with the mock signal.

Keys written (all read via np.load(...)['key'] or .item() in the analysis
script):
    spectra     (Nspectra, lenfft)  raw power spectra, arb. units
    lstvec      (Nspectra,)         local apparent sidereal time, hours
    datevec     (Nspectra, 3)       [year, month, day]
    timevec     (Nspectra, 3)       [hour, minute, second]
    mjdvec      (Nspectra,)         Modified Julian Date
    ravec       (Nspectra,)         RA, deg (ICRS)
    decvec      (Nspectra,)         Dec, deg (ICRS)
    lvec        (Nspectra,)         Galactic longitude, deg
    bvec        (Nspectra,)         Galactic latitude, deg
    azvec       (Nspectra,)         Azimuth, deg
    elvec       (Nspectra,)         Elevation, deg
    lenfft      scalar (0-d array)  FFT length per spectrum
    sample_rate scalar              Hz
    Tint        scalar              integration time per stored spectrum, sec
    NFFTave     scalar              number of raw FFTs averaged per spectrum
    freq        scalar              IF center frequency, Hz
    gain        scalar              SDR gain setting
"""

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, LSR

rng = np.random.default_rng(7)

# ----------------------------------------------------------------
# Observing location (matches script: Ithaca, NY)
# ----------------------------------------------------------------
Ithaca_latitude = 42 + 26.0 / 60
Ithaca_longitude = -(76.0 + 28.0 / 60)
observing_location = EarthLocation(
    lat=str(Ithaca_latitude), lon=str(Ithaca_longitude), height=300 * u.m
)

# Sun's velocity w.r.t. the LSR (same source the analysis script uses)
vsun_uvw = LSR().v_bary

c_kms = 299792.458        # km/s
frest_mhz = 1420.405752   # HI rest frequency, MHz

# ----------------------------------------------------------------
# Acquisition parameters (mirrors real A4410 lime file naming conventions)
# ----------------------------------------------------------------
LO_MHZ = 1160.0              # local oscillator, matches script default
IF_CENTERFREQ_HZ = 260.0e6   # -> RF center = 1160 + 260 = 1420 MHz exactly
SAMPLE_RATE_HZ = 10.0e6      # 10 MHz -> +/-5 MHz baseband span
LENFFT = 4096
NFFTAVE = 17578
GAIN = 10

Tint = NFFTAVE * LENFFT / SAMPLE_RATE_HZ  # seconds per stored spectrum (~7.2 s)

N_SPECTRA = 30000  # ~ Tint * 500 ~ 1 hour scan

# Fixed pointing -- this IS the drift scan: telescope parked, sky drifts by
AZ_DEG = 179.0
EL_DEG = 46.0

# ----------------------------------------------------------------
# Time axis
# ----------------------------------------------------------------
mjd_start = Time("2025-06-15T04:00:00", scale="utc").mjd
mjdvec = mjd_start + np.arange(N_SPECTRA) * (Tint / 86400.0)
tvec = Time(mjdvec, format="mjd", scale="utc")

isot = tvec.isot  # e.g. '2025-06-15T04:00:07.199'
datevec = np.array([[int(s[0:4]), int(s[5:7]), int(s[8:10])] for s in isot])
timevec = np.array(
    [[int(s[11:13]), int(s[14:16]), int(round(float(s[17:])))] for s in isot]
)

lstvec = tvec.sidereal_time("apparent", longitude=Ithaca_longitude * u.deg).hour

azvec = np.full(N_SPECTRA, AZ_DEG)
elvec = np.full(N_SPECTRA, EL_DEG)

# ----------------------------------------------------------------
# Astrometry: fixed Az/El -> RA/Dec, Galactic l/b, and radial velocity
# to the LSR, computed exactly as read_Lime_multispec_A4410.py does
# ----------------------------------------------------------------
directionvec = SkyCoord(
    az=azvec * u.deg, alt=elvec * u.deg, frame="altaz",
    obstime=tvec, location=observing_location,
)

radec = directionvec.icrs
ravec = radec.ra.deg
decvec = radec.dec.deg

galactic = directionvec.galactic
lvec = galactic.l.deg.copy()
bvec = galactic.b.deg
lvec[lvec > 180] -= 360

nhatvec = directionvec.galactic.cartesian
vlsr_radial_vec = nhatvec.dot(vsun_uvw)
barycorr_vec = directionvec.radial_velocity_correction()
vrlsrvec = vlsr_radial_vec.value
vrbaryvec = barycorr_vec.to(u.km / u.s).value
vradvec = vrlsrvec + vrbaryvec  # km/s, same convention as analysis script

fdopp_mhz = frest_mhz * (1.0 + vradvec / c_kms)   # observed HI freq, MHz
RF_center_mhz = LO_MHZ + IF_CENTERFREQ_HZ / 1e6   # = 1420.0 MHz
line_center_baseband_mhz = fdopp_mhz - RF_center_mhz  # where line falls in baseband

# ----------------------------------------------------------------
# Frequency axis (same construction as the analysis script)
# ----------------------------------------------------------------
f_baseband_vec = np.fft.fftshift(np.fft.fftfreq(LENFFT, 1e6 / SAMPLE_RATE_HZ))  # MHz

# ----------------------------------------------------------------
# Build the mock spectra
# ----------------------------------------------------------------
NOISE_LEVEL = 1.0
CONTINUUM = 8.0                 # flat receiver/bandpass baseline level
BANDPASS_RIPPLE_AMP = 0.6       # smooth slow ripple across the band
HI_LINEWIDTH_MHZ = 0.10         # ~20 km/s-ish velocity width
HI_PEAK_AMP = 3.5                # emission strength at b = 0
HI_LAT_SCALE_DEG = 25.0          # falloff of HI signal with |Galactic latitude|
DC_SPIKE_AMP = 15.0              # artifact spike at baseband f=0, as real SDR data show

spectra = np.empty((N_SPECTRA, LENFFT))

# smooth bandpass shape shared by all spectra (receiver gain vs. frequency)
bandpass_shape = CONTINUUM + BANDPASS_RIPPLE_AMP * np.cos(
    2 * np.pi * f_baseband_vec / (f_baseband_vec[-1] - f_baseband_vec[0]) * 1.5
)

for i in range(N_SPECTRA):
    noise = rng.normal(loc=0.0, scale=NOISE_LEVEL, size=LENFFT)
    spec = bandpass_shape + noise

    hi_amp = HI_PEAK_AMP * np.exp(-abs(bvec[i]) / HI_LAT_SCALE_DEG)
    hi_line = hi_amp * np.exp(
        -0.5 * ((f_baseband_vec - line_center_baseband_mhz[i]) / HI_LINEWIDTH_MHZ) ** 2
    )
    spec += hi_line

    # keep everything positive (these are "power" spectra)
    spec = np.clip(spec, 0.05, None)
    spectra[i] = spec

# DC/zero-baseband spike artifact (the analysis script explicitly patches this)
indfzero = np.abs(f_baseband_vec).argmin()
spectra[:, indfzero] += DC_SPIKE_AMP

# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------
outfile = "/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/data/60_hour_scan_mjd_%d.npz" % int(mjd_start)

np.savez(
    outfile,
    spectra=spectra,
    lstvec=lstvec,
    datevec=datevec,
    timevec=timevec,
    mjdvec=mjdvec,
    ravec=ravec,
    decvec=decvec,
    lvec=lvec,
    bvec=bvec,
    azvec=azvec,
    elvec=elvec,
    lenfft=np.array(LENFFT),
    sample_rate=np.array(SAMPLE_RATE_HZ),
    Tint=np.array(Tint),
    NFFTave=np.array(NFFTAVE),
    freq=np.array(IF_CENTERFREQ_HZ),
    gain=np.array(GAIN),
)

print("Wrote:", outfile)
print("Nspectra:", N_SPECTRA, " lenfft:", LENFFT, " Tint (s):", Tint)
print("RF center (MHz):", RF_center_mhz)
print("baseband span (MHz): %.3f to %.3f" % (f_baseband_vec[0], f_baseband_vec[-1]))
print("HI line baseband center range (MHz): %.4f to %.4f" %
      (line_center_baseband_mhz.min(), line_center_baseband_mhz.max()))
print("l range (deg): %.2f to %.2f" % (lvec.min(), lvec.max()))
print("b range (deg): %.2f to %.2f" % (bvec.min(), bvec.max()))