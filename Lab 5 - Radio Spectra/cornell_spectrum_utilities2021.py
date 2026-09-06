# 2021 Aug 27
# Utilities for spectra obtained for A4410 and A6525 lab
# 'choose' functions copied from rspecdrift2020.py

import numpy as np

from matplotlib.pyplot import ginput


def choose_time_range_for_spectrum_plot():
    """
    Prompts for choosing a time range on the plot window.
    Needs clicks on two time points in increasing time (or index) order.
    """

    happy=0
    while not(happy):
       print( "  Select time range for plotting a time slice")
       print( "  Use cursor to select 2 points per, earlier time first")
       print( "  First: click on plot window, then click on two time points in increasing time order")
       xxx = ginput(2, timeout=0)
       t1 = xxx[0][1]
       t2 = xxx[1][1]
       happy = 1
       print( t1, t2)
    return t1, t2

def choose_frequencies_for_polynomial_fitting(frequencies, spectrum, polydeg=3, nranges=2):
    """
    Prompts for specified number (nranges) of frequency ranges 
    for fitting a polynomial of degree polydeg.  There are two frequency 
    values for each range.

    It fits the input spectrum and returns the evaluated fitted polynomial.

    Input:
        frequencies = vector of frequencies for the input spectrum
        spectrum = input spectrum
        polydeg = order of polynomial to be fitted [3]
        nranges = number of frequency ranges to use [2]

    Output:
        franges = array with frequency ranges (2 x nranges) 
        ffit = vector of frequencies used for fit 
        sfit = vector of spectral values used for fit 
        params2 = parameters of fit 
        poly2 = evaluated polynomial fit

    """
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
         xxx = ginput(2, timeout=0)
         franges[nr,0] = xxx[0][0]
         franges[nr,1] = xxx[1][0]
       happy = 1
       print( franges)
    ffit=[]
    sfit=[]
    ffit=np.array(ffit)
    sfit=np.array(sfit)

    # nb: the nch, f loop can probably be replaced with usage of 'where' command
    for nch, f in enumerate(frequencies):
        for nr in range(nranges):
            #print "   ", franges[nr,0],  franges[nr,1]
            if franges[nr,0] <= f and f <= franges[nr,1]:
               ffit = np.append(ffit, f)
               sfit = np.append(sfit, spectrum[nch])

    params2 = np.polyfit(ffit, sfit, polydeg)
    poly2 = np.polyval(params2, frequencies)

    return franges, ffit, sfit, params2, poly2

def a4410_rtel():
    """
    Returns latitude, longitude, and altitude of 3.8 m telescope on top of
    the Space Sciences Building.

    Longitude, Latitude from topo map as read in 1990s.
    Confirmed 2012 Dec 18 using http://weather.gladstonefamily.net/site/KITH
    Can see antenna on google maps: Altitude given as 263 meters
    (from NED = National Elevation Dataset, http://ned.usgs.gov/)

    Going directly to http://viewer.nationalmap.gov/viewer/ and
    clicking on antenna on top of the SSB:

    Spot Elevation
    Feet: 865
    Meters: 264
    Source: NED 1/3rd arc-second: Eastern United States
    DD: 42.44879 -76.48119
    DMS: 42deg 26 arcmin 55.643 arcsec  N     76 deg 28 arcmin 52.295 arcsec W
    UTM: 18 378196 4700670
    USNG: 18T UN 78195 00669 (NAD 83)
    MGRS: 18TUN7819500669
    """

    RtelLongitude = -76.48122
    RtelLatitude = 42.44855
    RtelAltitude = 264.

    return RtelLatitude, RtelLongitude, RtelAltitude
