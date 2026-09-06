# 2025 Feb 23    Streamlined for HVC mapping project, 
#                added spectrum_vs_rf and zoomed avg. 
#                spectrum with bestf fit plots

# 2024 Dec 31-2025 Jan02    
#                enact total power plotting, add aitoff frame,  
#                change labeling to use data start date/time
#                now can identify lime vs airspy data.
# 2023 Nov 28    tweak
# 2022 Dec 8     Streamlined for A4410 class
# 2022 Nov 28    Some tweaks to handle new data sets
# 2021 Aug 27    Replace radial velocity loop calculation with vector calls
#                Include import of new cornell_spectrum_utilities2021.py
# 2021 June 17   Add nprocess to calling args (JMC)

# Initially: read_Lime_multispec_A4410.py by JMC 2019 Dec 4

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter

import argparse

import sys
import datetime 
from datetime import date

from astropy import constants
c = constants.c.cgs.value
c_kms = c / 1e5

import warnings
warnings.filterwarnings("ignore")

from astropy.utils.exceptions import AstropyWarning

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, LSR
from astropy.coordinates.representation import UnitSphericalRepresentation

from astropy.utils import iers
from astropy.utils.iers import conf
iers.conf.iers_auto_url=iers.IERS_B_URL

conf.auto_max_age = None
iers.conf.auto_download = True

import cornell_spectrum_utilities2021 as csu

Ithaca_latitude = 42 + 26./60                           # degrees
Ithaca_longitude = -(76.+28./60)                        # degrees

# Conversion factors
Hz2MHz = 1e6
MHz2GHz = 1e3
sec_in_day = 86400
pi = np.pi

script_name = sys.argv[0]
basename = script_name.split('.')[0]
now = datetime.datetime.now()

# Set up observing location object
observing_location = EarthLocation(
    lat=str(Ithaca_latitude), 
    lon=str(Ithaca_longitude), 
    height=300*u.m)                # NB height is a guess

# Components of the Sun's velocity with respect to the LSR

vsun_uvw = LSR().v_bary
U,V,W = vsun_uvw._xyz.value[:]         # km/s

def choose_time_range_for_spectrum_plot():

    happy=0
    while not(happy):
       print( "  Select time range for plotting a time slice")
       print( "  Use cursor to select 2 points per, earlier time first")
       print( "  First: click on plot window, then click on two time points in increasing time order")
       xxx = plt.ginput(2, timeout=0)
       t1 = xxx[0][1]
       t2 = xxx[1][1]
       happy = 1
       print( t1, t2)

    return t1, t2

def choose_frequencies(frequencies, spectrum, polydeg=3, nranges=2):

    # choose frequencies to use for fitting

    franges = np.zeros(shape=(nranges,2))      # pairs of frequencies
    happy=0
    while not(happy):
       print( "  Select ", nranges,  " frequency ranges to use in polynomial fit")
       print( "  Use cursor to select 2 points per range, left-hand point first")
       print( "  Frequency ranges should not overlap")
       print( "  Avoid obvious spectral lines")
       print( "  First: click on plot window:")
       for nr in range(nranges):
         print( "select frequency range #", nr, " starting at low frequencies")
         xxx = plt.ginput(2, timeout=0)
         franges[nr,0] = xxx[0][0]
         franges[nr,1] = xxx[1][0]
       happy = 1
       print( franges)
    ffit=[]
    sfit=[]
    ffit=np.array(ffit)
    sfit=np.array(sfit)
    for nch, f in enumerate(frequencies):
        for nr in range(nranges):
            #print "   ", franges[nr,0],  franges[nr,1]
            if franges[nr,0] <= f and f <= franges[nr,1]:
               ffit = np.append(ffit, f)
               sfit = np.append(sfit, spectrum[nch])

    #plot(ffit, sfit, 'go', markersize=3)
    params2 = np.polyfit(ffit, sfit, polydeg)
    poly2 = np.polyval(params2, frequencies)
    #plot(frequencies, poly2, 'k-', lw=3)

    return franges, ffit, sfit, params2, poly2

def plot_totalpower_and_raw_dynamic_spectrum(spectra, separate_plots=False):
    """
    Imshow plot of raw spectrum and plot of total power
    Name of function is a misnomer.  Need to fix.
    """
    smedian = np.median(spectra)
    smean = np.mean(spectra)
    smax = np.max(spectra)
    smin = np.min(spectra)
    tpower = np.average(spectra, axis=1)


    if separate_plots:
        title_label_ds = 'Raw dynamic spectrum ' + base_string + \
            '_' + startdate + '_' + starttime 

        raw_dynamic_spectrum_plotfile =  'Dynamic_spectrum_raw_data' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        fig = plt.figure()
        vmin = np.log10(0.8*smedian) 
        vmax = np.log10(1.5*smedian) 
        plt.imshow(np.log10(spectra), origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xlabel('Frequency index')
        plt.ylabel('Time index')
        plt.title(title_label_ds, fontsize=8)
        plt.suptitle('Input file = ' + basenpzfile, fontsize=7)
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{raw_dynamic_spectrum_plotfile}")
        #plt.show()
        

        
        title_label_tp = 'Total power time series_' + base_string + \
            '_' + startdate + '_' + starttime 
        total_power_time_series_plotfile =  'Total_power_time_series_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        tpower = np.average(spectra, axis=1)
        fig = plt.figure()
        plt.plot(tvec_hr, tpower, '.', ms=2)
        plt.xlabel('Time (hr)')
        plt.ylabel('Total power (arb. units)')
        plt.title(title_label_tp, fontsize=10)
        plt.suptitle('Input file = ' + basenpzfile, fontsize=7)
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{total_power_time_series_plotfile}")
        #plt.show()

    # ------------------
    # Combined DS and TP
    # ------------------
    combined_ds_and_tp_plotfile =  'DS_and_TP_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
    freq = np.fft.fftshift(np.fft.fftfreq(lenfft, 1e6/sample_rate))
    extent=((freq[0], freq[-1], 0., tvec_hr[-1]))

    # find range of spectral amplitudes containing 99% of the values
    # to set the color scale:
    h1, b1 = np.histogram(
    np.log10(spectra.flatten()),bins=1000,density=True)
    c1 = np.cumsum(h1)
    c1 /= c1.max()
    vmin = b1[np.where(c1<=0.005)][-1]
    vmax = b1[np.where(c1<=0.995)][-1]
    
    plt.figure()

    # Dynamic spectrum
    title_label_ds = r'$\rm Raw \  dynamic\ spectrum \ \  (\log_{10})$'

    ax1 = plt.axes((0.1, 0.1, 0.50, 0.7))
    plt.imshow(np.log10(spectra),aspect='auto',
        extent=extent, vmin=vmin, vmax=vmax,
        origin='lower',interpolation='nearest')
    #cbar = plt.colorbar(label=r'$\rm log_{10} \ \ S(\nu)$')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    #plt.annotate(r'$\rm log_{10} \ \ S(\nu)$',
        #xy=(1.375, 0.5), xycoords='axes fraction',
        #ha='center', va='center', fontsize=9, rotation=-90)
    plt.xlabel(r'$\rm RF - 1420 \ \ (MHz)$')
    plt.ylabel(r'$\rm Time \ \ (hr)$')
    plt.tick_params(axis='both', labelsize=8)
    #plt.annotate(plotstamp, xy=(0.6, 0.85), xycoords='figure fraction',
        #ha='left', va='center', fontsize=5)
    plt.title(title_label_ds, fontsize=7, pad=15)
    plt.suptitle('Input file = ' + basenpzfile, fontsize=7, x=0.5, y=0.925)
    plt.annotate(r'$\rm Az = %5.1f \ \ \ \ \ El = %5.1f$'%(AZ, EL),
        xy=(0.275, 0.820), xycoords='figure fraction',
        ha='center', va='center', fontsize=6)

    # Total power
    ax2 = plt.axes((0.65, 0.1, 0.25, 0.7))
    plt.plot(tpower, tvec_hr, '-', lw=1.5)
    plt.axis(ymin=0, ymax=tvec_hr[-1])
    plt.tick_params(axis='y', labelleft=False)
    plt.tick_params(axis='x', labelsize=8)
    plt.xlabel(r'$\rm Total \ power \ \ (arb.\ units)$')
    plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{combined_ds_and_tp_plotfile}")
    #plt.show()
    return

def remove_spectral_bandpass(
    RFvec, spectrum, 
    franges = np.array([[1.41896839, 1.41952398], [1.42076118, 1.42159744]]),
    degfit = 5,     # degree of polynomial fit
    mfilt = 10,      # number of samples used in median filter
    medfilt = True, 
    doplot = False):
    """
    RFvec in GHz
    spectrum = 1D spectrum
    mfilt = size of median filter (number of samples in block used to calculate median)
    """

    if medfilt:
        smedian = median_filter(spectrum, size=(mfilt,))
        sproc = np.copy(smedian)
    else:
        sproc = np.copy(spectrum)

    Nsegs = np.shape(franges)[0] 
    specfit_segs = np.zeros(Nsegs, dtype=object)
    rffit_segs = np.zeros(Nsegs, dtype=object)

    for ns in range(Nsegs):
        f1 = franges[ns,0]
        f2 = franges[ns,1]
        #print(ns, f1, f2)
        inds = np.where((f1 < RFvec) & (RFvec < f2))
        specfit_segs[ns] = sproc[inds] 
        rffit_segs[ns] = RFvec[inds]
        if ns == 0:
            rffit = np.copy(rffit_segs[ns])
            specfit = np.copy(specfit_segs[ns])
        else:
            rffit = np.append(rffit, rffit_segs[ns])
            specfit = np.append(specfit, specfit_segs[ns]) 

    # spectrum over range of baseline fit
    indsspan = np.where((franges[:,0].min() < RFvec) & (RFvec < franges[:,1].max()))
    RFspan = RFvec[indsspan]
    specspan = sproc[indsspan]

    mparams = np.polyfit(rffit, specfit, degfit)
    mfit = np.polyval(mparams, rffit)
    mfitresids = (specfit-mfit) / mfit
    mfitspan = np.polyval(mparams, RFspan)

    diffspec = (specspan - mfitspan) / mfitspan

    if doplot:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        plt.subplots_adjust(left=0.15, bottom=0.15, hspace=0.15)
        plt.plot(RFvec, spectrum, '-', lw=2, alpha=0.5, label='raw spectrum')
        plt.plot(RFvec, smedian, 'k-', lw=2.5, label='median filtered')
        plt.plot(rffit, specfit, 'r.', ms=7, label = 'for baseline fit')
        plt.plot(RFspan, mfitspan, 'c-', lw=2, label = 'baseline fit')
        plt.ylabel('Spectrum  (au)', fontsize=14)
        plt.legend(loc=0, ncol=2)

        aa,bb,cc,dd = plt.axis()

        ax = fig.add_subplot(212) 
        plt.plot(RFspan, diffspec, 'k-', lw=2)
        plt.axis(xmin=aa, xmax=bb)
        plt.ylabel(r'$\rm (S - S_{base}) / S_{base} \ \ (au)$', 
            fontsize=12)
        plt.xlabel('RF  (MHz)', fontsize=18)
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/spectrum_vs_rf_{startdate}_{starttime}.png")
        #plt.show()

    return RFspan, specspan, diffspec

def plot_ds(fvec, tvec_hr, spectra, dolog=False, heading=''):
        # ---------------------------------------------------------------
        # Dynamic spectrum + l, b panels: full frequency range 
        # ---------------------------------------------------------------
        Nspectra, Nf = np.shape(spectra)
        #plotfile = 'dynamic_spectrum_lb_' + basename + '_' +  \
            #base_string + '_' + startdate + '_' + starttime + '.png'
        title_label = 'HI data   MJD ' + str(MJD) + '   ' +  str(Nspectra)  + ' spectra'  
        extent=((fvec[0], fvec[-1], tvec_hr[0],  tvec_hr[-1]))
        #raw_dynamic_spectrum_and_lb_panels_plotfile =  'raw_DS_and_l,b_panels' + base_string + '_' + startdate + \
            #'_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        # find range of spectral amplitudes containing 99% of the values
        # to set the color scale:
        if dolog:
            sproc = np.log10(spectra)
        else:
            sproc = spectra
        h1, b1 = np.histogram(sproc.flatten(),bins=1000,density=True)
        c1 = np.cumsum(h1)
        c1 /= c1.max()
        vmin = b1[np.where(c1<=0.005)][-1]
        vmax = b1[np.where(c1<=0.995)][-1]

        plt.figure()
        ax1 = plt.axes((0.1, 0.1, 0.35, 0.7))
        plt.imshow(sproc,aspect='auto',
            extent=extent, vmin=vmin, vmax=vmax,
            origin='lower',interpolation='nearest')
        cbar = plt.colorbar(label=r'$\rm log_{10} \ \ S(\nu)$')
        cbar.ax.tick_params(labelsize=8)
        #plt.annotate(r'$\rm log_{10} \ \ S(\nu)$',
            #xy=(1.375, 0.5), xycoords='axes fraction',
            #ha='center', va='center', fontsize=9, rotation=-90)
        plt.xlabel(r'$\rm RF - 1420 \ \ (MHz)$')
        plt.ylabel(r'$\rm Time \ \ (hr)$')
        plt.tick_params(axis='both', labelsize=8)
        plotstamp = basename + '  ' +  startdate + ' ' + nowtime
        plt.annotate(plotstamp, xy=(0.6, 0.85), xycoords='figure fraction',
            ha='left', va='center', fontsize=5)
        plt.title(title_label, fontsize=7, pad=15)
        plt.suptitle('Input file = ' + basenpzfile, fontsize=7)
        plt.annotate(r'$\rm Az = %5.1f \ \ \ \ \ El = %5.1f$'%(AZ, EL), 
            xy=(0.25, 0.820), xycoords='figure fraction',
            ha='center', va='center', fontsize=6)
        if heading != '':
            plt.annotate(heading, xy=(0.25, 0.90), xycoords='figure fraction',
            ha='center', va='center', fontsize=9)

        # Plot Doppler shifted rest frequency from Earth orbital motion and 
        # the Sun's LSR motion:
        frest = 1420.405752                             # MHz
        fdopp = frest *(1.+vradvec/c_kms)
        plt.plot((fdopp-1420)[0:Nspectra], tvec_hr, 'w', dashes=[10,5], lw=0.5, alpha=0.5)

        nsplot = np.shape(spectra)[0]

        # longitude
        ax2 = plt.axes((0.55, 0.1, 0.15, 0.7))
        #ax2 = plt.axes((0.50, 0.1, 0.15, 0.7))
        plt.plot(lvec, tvec_hr, '-', lw=1.5)
        plt.plot((0,0), (0,tvec_hr[-1]), '--', lw=1)
        plt.axis(xmin=-190, xmax=190, ymin=0, ymax=tvec_hr[-1])
        plt.tick_params(axis='y', labelleft=False)
        plt.tick_params(axis='x', labelsize=8)
        plt.xlabel(r'$\rm \ell \ \ (deg)$')
        plt.xticks((-180, 0, 180))

        # latitude
        ax3 = plt.axes((0.75, 0.1, 0.15, 0.7))
        #ax3 = plt.axes((0.70, 0.1, 0.15, 0.7))
        plt.plot(bvec, tvec_hr, '-', lw=1.5)
        plt.plot((0,0), (0,tvec_hr[-1]), '--', lw=1)
        plt.axis(xmin=-95, xmax=95, ymin=0, ymax=tvec_hr[-1])
        plt.tick_params(axis='y', labelleft=False)
        plt.tick_params(axis='x', labelsize=8)
        plt.xlabel(r'$\rm b  \ \ (deg)$')
        plt.xticks((-90, 0, 90))
        #plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{raw_dynamic_spectrum_and_lb_panels_plotfile}")
        #plt.show()

        return

def plot_ds_and_aitoff(fvec, tvec_hr, spectra, dolog=False, heading=''):
        # ---------------------------------------------------------------
        # Dynamic spectrum + l, b panels: full frequency range 
        # ---------------------------------------------------------------
        Nspectra, Nf = np.shape(spectra)
        #plotfile = 'dynamic_spectrum_lb_' + basename + '_' +  \
            #base_string + '_' + startdate + '_' + starttime + '.png'
        title_label = 'HI data   MJD ' + str(MJD) + '   ' +  str(Nspectra)  + ' spectra'  
        extent=((fvec[0], fvec[-1], tvec_hr[0],  tvec_hr[-1]))
        # find range of spectral amplitudes containing 99% of the values
        # to set the color scale:
        if dolog:
            sproc = np.log10(spectra)
        else:
            sproc = spectra
        h1, b1 = np.histogram(sproc.flatten(),bins=1000,density=True)
        c1 = np.cumsum(h1)
        c1 /= c1.max()
        vmin = b1[np.where(c1<=0.005)][-1]
        vmax = b1[np.where(c1<=0.995)][-1]

        plt.figure()
        #ax1 = plt.axes((0.1, 0.1, 0.35, 0.7))
        ax1 = plt.axes((0.1, 0.1, 0.50, 0.7))
        plt.imshow(sproc,aspect='auto',
            extent=extent, vmin=vmin, vmax=vmax,
            origin='lower',interpolation='nearest')
        cbar = plt.colorbar(label=r'$\rm log_{10} \ \ S(\nu)$')
        cbar.ax.tick_params(labelsize=8)
        #plt.annotate(r'$\rm log_{10} \ \ S(\nu)$',
            #xy=(1.375, 0.5), xycoords='axes fraction',
            #ha='center', va='center', fontsize=9, rotation=-90)
        plt.xlabel(r'$\rm RF - 1420 \ \ (MHz)$')
        plt.ylabel(r'$\rm Time \ \ (hr)$')
        plt.tick_params(axis='both', labelsize=8)
        plotstamp = basename + '  ' +  startdate + ' ' + nowtime
        plt.annotate(plotstamp, xy=(0.6, 0.85), xycoords='figure fraction',
            ha='left', va='center', fontsize=5)
        plt.title(title_label, fontsize=7, pad=15)
        plt.suptitle('Input file = ' + basenpzfile, fontsize=7)
        plt.annotate(r'$\rm Az = %5.1f \ \ \ \ \ El = %5.1f$'%(AZ, EL), 
            xy=(0.25, 0.820), xycoords='figure fraction',
            ha='center', va='center', fontsize=6)
        if heading != '':
            plt.annotate(heading, xy=(0.25, 0.90), xycoords='figure fraction',
            ha='center', va='center', fontsize=9)

        # Plot Doppler shifted rest frequency from Earth orbital motion and 
        # the Sun's LSR motion:
        frest = 1420.405752                             # MHz
        fdopp = frest *(1.+vradvec/c_kms)
        plt.plot((fdopp-1420)[0:Nspectra], tvec_hr, 'w', dashes=[10,5], lw=0.5, alpha=0.5)

        nsplot = np.shape(spectra)[0]

        # Aitoff display of Galactic coordinates

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='aitoff')
        ax2 = plt.axes((0.65, 0.4, 0.25, 0.5), projection='aitoff')
        plt.grid(True)
        plt.tick_params(labelbottom=False)
        plt.tick_params(labelleft=False)
        plt.text(0,np.deg2rad(-120),'Galactic l', ha='center', va='center')
        plt.ylabel('Galactic b')
        ax2.yaxis.set_label_position("right")
        #plt.ylabel('Galactic latitude (degrees)')

        # Galactic poles
        plt.text(0,np.deg2rad(82.5),'NGP', ha='center', va='center', fontsize=8)
        plt.text(0,np.deg2rad(-83),'SGP', ha='center', va='center', fontsize=8)
        lrad = np.deg2rad(lvec)
        brad = np.deg2rad(bvec)
        plt.plot(-lrad, brad, '.', ms=5)

        # Galactic equator
        plt.plot(-lrad, np.zeros(np.size(lrad)), 'k:', ms=2)
        plt.plot(np.linspace(-pi, pi, 100), np.zeros(100), 'k:', ms=2)

        # First data point
        plt.plot(-lrad[0], brad[0], 'r.', ms=6)
        ds_aitoff_plotfile =  'flattened_ds_aitoff_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{ds_aitoff_plotfile}")
        #plt.show()

        """
        # longitude
        ax3 = plt.axes((0.55, 0.1, 0.15, 0.7))
        #ax3 = plt.axes((0.50, 0.1, 0.15, 0.7))
        plt.plot(lvec, tvec_hr, '-', lw=1.5)
        plt.plot((0,0), (0,tvec_hr[-1]), '--', lw=1)
        plt.axis(xmin=-190, xmax=190, ymin=0, ymax=tvec_hr[-1])
        plt.tick_params(axis='y', labelleft=False)
        plt.tick_params(axis='x', labelsize=8)
        plt.xlabel(r'$\rm \ell \ \ (deg)$')
        plt.xticks((-180, 0, 180))

        # latitude
        ax4 = plt.axes((0.75, 0.1, 0.15, 0.7))
        ax4 = plt.axes((0.70, 0.1, 0.15, 0.7))
        plt.plot(bvec, tvec_hr, '-', lw=1.5)
        plt.plot((0,0), (0,tvec_hr[-1]), '--', lw=1)
        plt.axis(xmin=-95, xmax=95, ymin=0, ymax=tvec_hr[-1])
        plt.tick_params(axis='y', labelleft=False)
        plt.tick_params(axis='x', labelsize=8)
        plt.xlabel(r'$\rm b  \ \ (deg)$')
        plt.xticks((-90, 0, 90))
        plt.show()
        #plt.savefig(plotfile)
        """

        return

# =========================================================================

# Main

do_diagnostic_plots = False 
do_diagnostic_plots = True

LO_default = 1160                               # MHz

default_file = 'data_lime_multispec_AzEl_179_69_20201123_mjd_59176_LST_220344_4096_19836_260_8_g_30.npz'
default_file = 'data_lime_multispec_AzEl_179_46_20210317_mjd_59290_LST_022935_4096_19836_260_8_g_30.npz'
default_file = 'data_lime_multispec_AzEl_179_59_20210816_mjd_59442_LST_133621_4096_19836_260_8_g_30.npz'
default_file = 'data_lime_multispec_AzEl_179_63_20210810_mjd_59436_LST_130648_4096_19836_260_8_g_30.npz'

default_file = 'data_lime_multispec_AzEl_179_46_20210317_mjd_59290_LST_022935_4096_19836_260_8_g_30.npz'
default_file = 'data_lime_multispec_lb_8_2_20221123_mjd_59906_LST_175021_1024_24414_260_5_g_30.npz'
default_file = 'data_lime_multispec_AzEl_179_47_20221123_mjd_59906_LST_175021_2048_48828_260_10_g_30.npz'

default_file = 'data_lime_multispec_lb_352_21_20231127_mjd_60275_LST_191950_2048_12207_260_5_g_30.npz'

default_file = 'data_lime_multispec_lb_37_20_20231202_mjd_60280_LST_202017_1024_48828_260_5_g_20.npz'

default_file='data_lime_multispec_lb_18_46_20241022_mjd_60605_LST_154406_4096_12207_260_5_g_30.npz'

default_file = 'data_lime_multispec_lb_336_70_20241023_mjd_60606_LST_133608_4096_12207_260_5_g_30.npz'

default_file = '/Volumes/JMCBook2022/data_lime_multispec_lb_68_-33_20241213_mjd_60657_LST_215836_4096_17089_260_10_g_10.npz'

default_file = '/Volumes/JMCBook2022/data_lime_multispec_lb_101_-50_20250107_mjd_60682_LST_235900_4096_17578_260_10_g_10.npz'

default_file = '/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/data_lime_multispec_lb_mock_20250615_mjd_60841_LST_mock_4096_10_260_10_g_10_24hrs.npz'

default_file = '/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/data/60_hour_scan_mjd_60841.npz'

doplot = True

#=========================================================================

parser = argparse.ArgumentParser('Input file needs to include full path\n')

parser.add_argument('-lo', '--lo', type=float, default=LO_default,
    help='LO frequency [MHz]  [default=1160 MHz for A4410 dish antenna system]')

parser.add_argument('-f', '--npzfile', type=str, default=default_file,
    help='Input npz file from lime_multipec_A4410...py or airspy_multispec_A4410...py')

parser.add_argument('-n', '--nprocess', type=int, default=-1,
    help='Maximum number of spectra to process')

parser.add_argument('-w', '--write_short_file', action='store_true', 
    help='If True Writes a smaller version of input file if -n used')

args = parser.parse_args()

#=========================================================================

npzfile = args.npzfile
Maxproc = args.nprocess
dowrite = args.write_short_file
LO = args.lo

basenpzfile = npzfile.split('/')[-1]            # for plot labeling

# Determine which SDR used to obtain data (from file name):
if 'lime' in npzfile:
    sdr = 'lime'
elif 'airspy' in npzfile:
    sdr = 'airspy'
else:
    sdr = 'generic'

print('Will read npz file ', npzfile)
print('Load npz file and unpack')
xxx = np.load(npzfile)

allspectra = xxx['spectra']
lstvec = xxx['lstvec']
datevec = xxx['datevec']
timevec = xxx['timevec']
mjdvec = xxx['mjdvec']
ravec = xxx['ravec']
decvec = xxx['decvec']
lvec_in = xxx['lvec']
bvec_in  = xxx['bvec']
azvec = xxx['azvec']
elvec = xxx['elvec']
NFFT = lenfft = xxx['lenfft'].item()
sample_rate = xxx['sample_rate'].item()         # Hz
Tint = xxx['Tint'].item()
NFFTaveraged_per_spectrum = NFFTave = xxx['NFFTave'].item()
IF_centerfreq = centerfreq = xxx['freq'].item() # Hz
gain = xxx['gain'].item()


"""
# Could use this approach to extract from xxx
# but as above scalars need further extraction with '.item'
# so not doing this now.
# Extract contents of xxx
print('Input file keys:\n', xxx.files)

for key in xxx.keys():
   exec( key + ' = xxx[key]') 

# Rename spectra for clarity:
allspectra = copy(spectra)
"""

Nspectra_in_file  = np.shape(allspectra)[0]

# If data acq had to be aborted before finishing the actual number of good spectra in the 
# file is less than what shape(allspectra)[0] gives (because of the way
# memory mapping was used to store the spectra)
#
# Incomplete spectra are all zeros; ditto the 1D vectors.
# So look for first entry of zeros and truncate all arrays accordingly.


Nactual = np.size(np.where(mjdvec > 0))
if Nactual < Nspectra_in_file:
    print('Mitigating truncated file')
    Nspectra_in_file = Nactual
    allspectra = allspectra[:Nactual]
    lstvec = lstvec[:Nactual]
    datevec = datevec[:Nactual]
    timevec = timevec[:Nactual]
    mjdvec = mjdvec[:Nactual]
    ravec = ravec[:Nactual]
    decvec = decvec[:Nactual]
    lvec_in = lvec_in[:Nactual]
    bvec_in = bvec_in[:Nactual]
    azvec = azvec[:Nactual]
    elvec = elvec[:Nactual]


# Process Nspectra_in_file or a subset Maxproc if smaller (and positive)
if Maxproc > 0 and Nspectra_in_file > Maxproc:      # subset of file
    Nspectra = Maxproc
    spectra = allspectra[0:Nspectra]
    lvec = lvec_in[0:Nspectra]
    bvec = bvec_in[0:Nspectra]
    if dowrite:
        outfile =  npzfile.replace('.npz', '_short_' + '%d'%(Nspectra))
        np.savez(outfile, spectra=spectra, lstvec=lstvec[0:Nspectra], datevec=datevec[0:Nspectra], timevec=timevec[0:Nspectra], mjdvec=mjdvec[0:Nspectra], ravec=ravec[0:Nspectra], decvec=decvec[0:Nspectra], lvec=lvec[0:Nspectra], bvec=bvec[0:Nspectra], azvec=azvec[0:Nspectra], elvec=elvec[0:Nspectra], lenfft=lenfft, NFFTave=NFFTave, sample_rate=sample_rate, Tint=Tint, freq=centerfreq, gain=gain)
else:
    Nspectra = Nspectra_in_file
    spectra = allspectra
    lvec = lvec_in
    bvec = bvec_in

print('Calculate frequencies')

f_baseband_vec = np.fft.fftshift(np.fft.fftfreq(NFFT, Hz2MHz/sample_rate)) # MHz
RFvec = (LO + IF_centerfreq / Hz2MHz +  f_baseband_vec) / MHz2GHz       # GHz
RFcenter = (LO + (IF_centerfreq / Hz2MHz)) / MHz2GHz                    # GHz 

freq = np.fft.fftshift(np.fft.fftfreq(lenfft, Hz2MHz/sample_rate))

# fix zero baseband frequency value (replace spike with mean of adjacent values)
indfzero = abs(f_baseband_vec).argmin()
spectra[:, indfzero] = 0.5*(spectra[:, indfzero-1] + spectra[:, indfzero+1])

# put longitudes into [-180, 180] range:
lvec[np.where(lvec>180)] -= 360

# Times:
tvec_hr = (mjdvec[0:Nspectra]-mjdvec[0]) * 24           # time in hours

AZ = azvec[0]
EL = elvec[0]
MJD = mjdvec[0].astype(int)
nowdate, nowtime  = str(now).split(' ')[0:]
nowtime = nowtime.split('.')[0]

startdate = '%4d-%2d-%2d'%(datevec[0][0], datevec[0][1], datevec[0][2])
enddate = '%4d-%2d-%2d'%(datevec[-1][0], datevec[-1][1], datevec[-1][2])

starttime = ('%2d:%2d:%2d'%(timevec[0][0],timevec[0][1],timevec[0][2])).replace(' ','0')
endtime = ('%2d:%2d:%2d'%(timevec[-1][0],timevec[-1][1],timevec[-1][2])).replace(' ','0')

plotstamp = basename + '  ' +  startdate + ' ' + starttime 

# Calculate radial velocity to LSR
print('Calculate radial velocities (this can be slow for a large file):')

# Vector calculations: only for the requested number of time steps (Nspectra)
mjdvec_req = mjdvec[0:Nspectra]
azvec_req = azvec[0:Nspectra]
elvec_req = elvec[0:Nspectra]

tvec_req = Time(mjdvec_req, format='mjd')
directionvec = SkyCoord(az=azvec_req*u.deg, alt=elvec_req*u.deg, frame='altaz', 
    obstime=tvec_req, location=observing_location)
nhatvec = directionvec.galactic.cartesian
vlsr_radial_vec = nhatvec.dot(vsun_uvw)
barycorr_vec = directionvec.radial_velocity_correction()
bb_vec = barycorr_vec.to(u.km/u.s)
vrlsrvec = vlsr_radial_vec.value
vrbaryvec = bb_vec.value
vradvec = vrlsrvec + vrbaryvec

mean_spec = np.average(spectra, axis=0)

base_string = str(lenfft) + '_' + str(NFFTave) + '_' + str(int(IF_centerfreq/1.e6)) \
    + '_' + str(int(sample_rate/1.e6)) + '_g_' + str(int(gain))

if Nspectra > 1 and doplot == True:

    if do_diagnostic_plots:
        print('Plotting dynamic spectrum using %d spectra'%(Nspectra))
        # ---------------------------------------------------------------
        # Imshow plot dynamic spectrum if Nspectra > 1
        raw_dynamic_spectrum_plotfile =  'Dynamic_spectrum_raw_data' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        title_label = 'dynamic_spectrum_' + base_string + \
            '_' + startdate + '_' + starttime + '.png'
        #freq = np.fft.fftshift(np.fft.fftfreq(lenfft, 1e6/sample_rate))
        #extent=((freq[0], freq[-1], 0., Tint*Nspectra)) 
        extent=((f_baseband_vec[0], f_baseband_vec[-1], 0., Tint*Nspectra)) 
        # find range of spectral amplitudes containing 99% of the values
        # to set the color scale:
        h1, b1 = np.histogram(
            np.log10(spectra.flatten()),bins=1000,density=True)
        c1 = np.cumsum(h1)
        c1 /= c1.max()
        vmin = b1[np.where(c1<=0.005)][-1]
        vmax = b1[np.where(c1<=0.995)][-1]

        plt.figure()
        plt.imshow(np.log10(spectra),aspect='auto',
            extent=extent, vmin=vmin, vmax=vmax,
            origin='lower',interpolation='nearest')
        plt.colorbar()
        plt.xlabel(r'$\rm Frequency \ (MHz)$')
        plt.ylabel(r'$\rm Time \ (sec)$')
        plt.annotate(plotstamp, xy=(0.7, 0.02), xycoords='figure fraction', 
            ha='left', va='center', fontsize=5)
        plt.title(title_label, fontsize=9)
        plt.suptitle('Input file = ' + basenpzfile, fontsize=7)
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{raw_dynamic_spectrum_plotfile}")
        #plt.show()

        """
        PUT HERE:
        1. plot average spectrum
        2. calculate polynomial fit on average spectrum to define frequency ranges
           for flattening
        3. step through all spectra, calculate fit, subtract and divide by fit
           to get flattened spectra
        Example call:   (note spectrumQ1 is calculated much later so need a
        raw average spectrum first)
        franges, ffit, sfit, params2, poly2 = csu.choose_frequencies_for_polynomial_fitting(freq, spectrumQ1)
  
        """
        print('Plotting raw average spectrum')
        # ---------------------------------------------------------------
        # Plot averaged spectrum
        # ---------------------------------------------------------------
        total_power_time_series_plotfile =  'Total_power_time_series_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        #freq = np.fft.fftshift(np.fft.fftfreq(lenfft, 1e6/sample_rate))
        plt.figure()
        plt.plot(f_baseband_vec, mean_spec)
        plt.yscale('log')
        plt.xlabel(r'$\rm Frequency \ (MHz)$')
        plt.ylabel(r'$\rm \log_{10} \ Spectrum \ (arbitrary\ units)$')
        plt.annotate(plotstamp, xy=(0.7, 0.02), xycoords='figure fraction', 
            ha='left', va='center', fontsize=5)
        plt.title(total_power_time_series_plotfile, fontsize=9)
        plt.suptitle('Input file = ' + basenpzfile, fontsize=7)
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{total_power_time_series_plotfile}")
        #plt.show()
        

        print('Plotting dynamic spectrum with l,b panels')
        #----------------------------------------------------------------
        # Dynamic spectrum + l, b panels: full frequency range 
        # ---------------------------------------------------------------
        heading = 'raw spectrum with l,b panels'
        plot_ds(RFvec*1000-1420, tvec_hr, spectra, dolog=True, heading=heading)
        raw_dynamic_spectrum_and_lb_panels_plotfile =  'raw_DS_and_l,b_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{raw_dynamic_spectrum_and_lb_panels_plotfile}")

        print('Plotting zoomed dynamic spectrum with l,b panels')
        # --------------------------------
        # Create array with zoomed spectra
        # --------------------------------
        fmin, fmax = -0.5, 1.5

        indsf = np.squeeze(np.where((fmin <= f_baseband_vec) & (f_baseband_vec <= fmax)))
        fzoom = f_baseband_vec[indsf] 
        szoom= spectra[:, indsf]
        avezoom = np.average(szoom, axis=0)

        # ---------------------------------------------------------------
        # Dynamic spectrum + l, b panels: zoomed in frequency range 
        # NB: zoom is done by using plt.axis() not with new array 
        # ---------------------------------------------------------------
        zoomed_dynamic_spectrum_lb_plotfile =  'zoomed_dynamic_spectrum_lb' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        heading = 'Zoomed RF range'
        title_label = 'HI data   MJD ' + str(MJD) + '   ' +  str(Nspectra)  + ' spectra'  
        plot_ds(RFvec[indsf]*1000-1420, tvec_hr, spectra[:,indsf], dolog=False, heading=heading)
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{zoomed_dynamic_spectrum_lb_plotfile}")

        print('Plotting zoomed average spectrum')
        # Plot zoomed average spectrum 

        fig = plt.figure()
        zoomed_average_spectrum_plotfile =  'zoomed_average_spectrum_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        ax = fig.add_subplot(111)
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.plot(fzoom, avezoom, 'b-')
        plt.xlabel(r'$\rm Baseband \ frequency \ \ (MHz)$', fontsize=15)
        plt.ylabel(r'$\rm Spectrum \ \ \ (arb\ units)$', fontsize=15)
        plt.suptitle('Input file = ' + basenpzfile, fontsize=7)
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{zoomed_average_spectrum_plotfile}")
        #plt.show()

        print('Plotting fitted zoomed average spectrum')

        # Plot zoomed average spectrum with linear fit

        fig = plt.figure()
        fitted_zoomed_average_spectrum_plotfile =  'zoomed_average_spectrum_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
        ax = fig.add_subplot(111)
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.plot(fzoom, avezoom, 'b-', label='Data')

        coeffs = np.polyfit(fzoom, avezoom, 5)  # fits a n-degree polynomial 
        fit_line = np.polyval(coeffs, fzoom)  

        plt.plot(fzoom, fit_line, 'r--', label=f'Linear Fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
        plt.xlabel(r'$\rm Baseband \ frequency \ \ (MHz)$', fontsize=15)
        plt.ylabel(r'$\rm Spectrum \ \ \ (arb\ units)$', fontsize=15)
        plt.suptitle('Input file = ' + basenpzfile, fontsize=7)
        plt.legend()
        plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{fitted_zoomed_average_spectrum_plotfile}")
        #plt.show()

        print('Plotting raw dynamic spectrum and total power time series')
        # ----------------------------
        # Plot total power time series
        # ----------------------------
        plot_totalpower_and_raw_dynamic_spectrum(spectra)

        # End diagnostic plot section

    # -------------------------------------------------------
    # Calculate median spectrum and remove spectral baselines
    # -------------------------------------------------------
    degfit = 5              # order of polynomial
    mfilt = 10              # length of median filter (in samples)

    print('Calculating and plotting flattened DS and total power time series')
    # Average spectrum baseline removal as demo
    specave = np.average(allspectra, axis=0)
    RFspan, specspan, diffspec = \
        remove_spectral_bandpass(RFvec, specave, medfilt=True, doplot=True)

    for ns, spectrum in enumerate(spectra):
        RFspan, specspan, diffspec = remove_spectral_bandpass(
           RFvec, spectrum, 
           franges = np.array([[1.41896839, 1.41952398], [1.42076118, 1.42159744]]),
           degfit=degfit,       # degree of polynomial fit
           mfilt=mfilt,         # number of samples used in median filter
           medfilt=True, doplot=False)
        if ns == 0:
            dspectra = np.zeros((Nspectra, np.size(diffspec)))
        dspectra[ns] = diffspec

    print('Plotting flattened dynamic spectra and l,b time series')
    heading = r'$\rm Flattened\ using \ %d-deg \ polynomial$'%(degfit)
    plot_ds(RFspan*1000-1420, tvec_hr, dspectra, dolog=False, heading=heading)
    flattened_ds_lb_plotfile =  'flattened_ds_lb_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
    plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{flattened_ds_lb_plotfile}")

    print('Plotting flattened dynamic spectra and aitoff display')
    plot_ds_and_aitoff(RFspan*1000-1420, tvec_hr, dspectra, dolog=False, heading=heading)
    ds_aitoff_plotfile =  'flattened_ds_aitoff_' + base_string + '_' + startdate + \
            '_' + starttime + '_Nspec_%d'%(Nspectra) + '.png'
    plt.savefig(f"/Users/Djslime07/ASTRO-4410-FA26-mts246/Lab 5 - Radio Spectra/plots/{ds_aitoff_plotfile}")
    

"""
# Next: fold spectra vs time if there are multiple cycles
# Quick test 2025 Jan 15:
# Need to finish this: periodicity easily seen in time series
# single frequency slice shows lots of harmonics but it's noisy
indrest = abs(RFspan-1.42040).argmin()
dsslice = dspectra[:,586]
#dsslice -= dsslice.mean()
ps_dsslice = fftshift(abs(fft.fft(dsslice))**2)
Nt = np.size(tvec_hr)
dt = tvec_hr[1]-tvec_hr[0]            # time sampling in hr
df = 1 / tvec_hr[-1]            # frequency sampling in cycles/hr
fNy = 1 / (2*dt)                # Nyquist frequency in cycles/hr 
fvec = np.linspace(-fNy, fNy, Nt) 
f24 = 1 / 24.
i24 = f24 / df
# want to find fundamental frequency and use it for folding
fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(tvec_hr, dsslice)

ax = fig.add_subplot(212)
fplotmax = 0.5                  # cycles/hr
indsplot = np.where((fvec > 0) & (fvec <= fplotmax)) 
indsplot = indsplot[0] + 1

plt.plot(fvec[indsplot], ps_dsslice[indsplot])
plt.yscale('log')
plt.show()

"""
# buffer
#print("Starting buffer plot")
#plt.figure()
#plt.subplot(projection="aitoff")
#plt.title("Aitoff")
#plt.grid(True)
#plt.show()
#print("Ending buffer plot")


input('hit return')
plt.close('all')