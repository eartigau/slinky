from ast import mod
from sre_compile import SRE_FLAG_LOCALE
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from tqdm import tqdm
from scipy.optimize import curve_fit

# Speed of light in m/s
c = 299792458

def gauss_floor(x, a, sigma):
    """Gaussian function with zero floor and centered at zero."""
    return a * np.exp(-0.5 * (x  / sigma) ** 2)

def covariance_vs_distance(x, y, xp, dx_tol=None):
    """
    Compute covariance of y as a function of distance in x for non-uniformly sampled data.
    
    This computes an autocorrelation-like function by binning point pairs by their 
    x-distance separation and computing the covariance for each bin.

    Parameters:
    -----------
    x : array-like
        X coordinates (non-uniform sampling)
    y : array-like
        Y coordinates
    xp : array-like
        Grid of distances in x to compute covariance at
    dx_tol : float, optional
        Tolerance for distance matching. If None, uses half the median spacing in xp
        
    Returns:
    --------
    xp : array
        Grid of distances (same as input)
    cov_vals : array
        Covariance values at each distance in xp
    n_pairs : array
        Number of point pairs in each distance bin
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    xp = np.asarray(xp)
    
    # Remove NaN values from x and y
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    
    # Mean subtraction for covariance computation
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    if dx_tol is None:
        # Use half the median spacing as tolerance
        dx_tol = np.median(np.diff(np.sort(xp))) / 2
    
    # Maximum distance we care about - only compute pairs up to this distance
    max_dist = np.max(xp) + dx_tol
    
    # Sort by x coordinate for efficient distance cutoff
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_centered_sorted = y_centered[sort_idx]
    
    cov_vals = np.zeros(len(xp))
    n_pairs = np.zeros(len(xp), dtype=int)
    
    # Loop with progress bar - only compute pairs up to max_dist
    for i in tqdm(range(len(x_sorted)), desc="Computing covariance"):
        # Find upper limit: where distance exceeds max_dist
        j_end = np.searchsorted(x_sorted, x_sorted[i] + max_dist, side='right')
        
        j_start = i + 1
        if j_start < j_end:
            # Vectorized distance computation for nearby points only
            dx = x_sorted[j_start:j_end] - x_sorted[i]
            y_prod = y_centered_sorted[i] * y_centered_sorted[j_start:j_end]
            
            # Find closest bin for each distance (vectorized)
            bin_idx = np.argmin(np.abs(xp[:, np.newaxis] - dx), axis=0)
            
            # Check which ones are within tolerance
            within_tol = np.abs(xp[bin_idx] - dx) <= dx_tol
            
            # Accumulate only those within tolerance
            valid_bins = bin_idx[within_tol]
            valid_prods = y_prod[within_tol]
            
            np.add.at(cov_vals, valid_bins, valid_prods)
            np.add.at(n_pairs, valid_bins, 1)
    
    # Normalize by number of pairs
    valid = n_pairs > 0
    cov_vals[valid] /= n_pairs[valid]
    cov_vals[~valid] = np.nan
    
    return xp, cov_vals, n_pairs


def slinky_fit(x, y, yerr, wslinky=1e-1):
    """
    Project data points onto a regular grid using a Gaussian weight.

    :param x: The x values for which we have data and errors
    :param y: The y values for which we have data and errors
    :param yerr: The error on y
    :param wslinky: The e-width of the Gaussian kernel
    :param xmin: The starting point of the grid
    :param xmax: The end point of the grid
    :param npts: The number of points in the grid
    :return: The x and y values of the grid at which we have projected the data
    """

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)

    x = np.array(x)[valid]
    y = np.array(y)[valid]
    yerr = np.array(yerr)[valid]

    xmin = np.min(x)
    xmax = np.max(x)

    # the caracterisic length is the FWHM/2.355, we want >3 points per FWHM
    # so by using 2*wslinky, we have ~3.5 points per FWHM
    npts = int( 2*(xmax - xmin) / wslinky )

    # Create a grid of x values
    xv = np.linspace(xmin, xmax, npts)
    # Initialize weights and y values for the grid
    weights = np.full(npts, 1e-12)
    yv = np.zeros(npts)

    xvbis = xv / wslinky
    xbis = x / wslinky
    # Loop over each data point
    for i in tqdm(range(len(x)), leave=False):
        # Calculate the distance between the grid points and the data point
        dd = xvbis - xbis[i]
        g = np.abs(dd) < 10

        dd2 = dd[g]

        # Calculate the weight of the data point
        w2 = np.exp(-0.5 * dd2 ** 2) / yerr[i] ** 2
        # Add the weight to the grid weights
        weights[g] += w2
        # Add the weighted y value to the grid y values
        yv[g] += w2 * y[i]
    # Normalize the y values by the weights
    yv /= weights

    return xv, yv


# Load FITS tables containing spectral line data from three fibers

instrument = 'SPIRou'
# demo for SPIRou--

if instrument == 'SPIRou':
    tbl_A = Table.read('3444961B5Da_pp_e2dsff_A_waveref_fplines_A.fits')
elif instrument == 'NIRPS':
    tbl_A = Table.read('C7A164F31A_pp_e2dsff_A_waveref_fplines_A.fits')
else:
    raise ValueError('Unsupported instrument')

#tbl_C = Table.read('3444961B5Da_pp_e2dsff_C_waveref_fplines_C.fits')

# Extract wavelength reference, velocity shift, and pixel position for Fiber A
wave_A = np.array(tbl_A['WAVE_REF'])
# Calculate velocity shift: dv = (1 - wavelength_ref/wavelength_meas) * c
dv_A = (1-np.array(tbl_A['WAVE_REF'])/np.array(tbl_A['WAVE_MEAS']))*c
pix_A = tbl_A['PIXEL_REF']

# Compute covariance of velocity shift as a function of wavelength distance
# This tells us how correlated the velocity shifts are at different wavelength separations
space_cov_dv = np.linspace(0, 2, 100)**2
too_small =  np.nanmedian(np.diff(wave_A))>space_cov_dv
space_cov_dv = space_cov_dv[~too_small]

cov_dv = covariance_vs_distance(wave_A, dv_A, space_cov_dv)

# in pcov, we force the zero point to zero and center to zero, we only fit amplitude and sigma
# Fit Gaussian to measured covariance with constraint that sigma must be positive
popt, pcov = curve_fit(gauss_floor, 
                       cov_dv[0], cov_dv[1], 
                       p0=[np.nanmax(cov_dv[1]), 1.0], 
                       bounds=([0, 0], [np.inf, np.inf]))

# Convert sigma to e-width (characteristic width of correlation)
ew_cov = popt[1]/np.sqrt(2)

print(f'Covariance e-width: {ew_cov} nm')

# Plot covariance and fitted Gaussian
plt.plot(cov_dv[0], cov_dv[1], '.', label='Covariance')
plt.plot(cov_dv[0], gauss_floor(cov_dv[0], *popt), '-', label='Gaussian Fit')
plt.xlabel('Wavelength Distance (nm)')
plt.ylabel('Covariance of Delta V (m/s)^2')
plt.title('Covariance of Delta V vs Wavelength Distance')
plt.legend()
plt.show()

order_A = tbl_A['ORDER']

# Apply wavelength-dependent smoothing using covariance length scale
slinky1 = slinky_fit(wave_A, dv_A, yerr=np.ones_like(dv_A)*10, wslinky=ew_cov)

# Fit splines to smoothed data for fine structure extraction
spl1 = ius(slinky1[0], slinky1[1], k=1, ext=0)

# Extract fine structure by subtracting smoothed component
dv_A_slinky = dv_A - spl1(wave_A)

# Create comprehensive 4-panel figure showing different processing stages
fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True, sharey=True)
for iord in np.unique(order_A):
    sel = order_A == iord
    # Panel 1: Original data with smoothing curves
    ax[0].plot(wave_A[sel], dv_A[sel], '.', label=f'Order {iord}')
    ax[0].plot(slinky1[0], slinky1[1], '-', color='grey', alpha=0.5)
    
    # Panel 2: Fine structure from original data
    ax[1].plot(wave_A[sel], dv_A_slinky[sel], '.', label=f'Order {iord}')

ax[0].set_ylabel('Delta V (m/s)')
ax[0].set_title('Fiber A: Delta V vs Wavelength')
ax[1].set_ylabel('Delta V - Slinky (m/s)')
plt.show()

# to apply to wavelength solution, we define a spline of the slinky
#
# slinky_spline = ius(slinky1[0], slinky1[1], k=2, ext=3)
#
# wave_corr = wave*(1+slinky_spline(wave_A) / c
# 
#. Compare HC positions relative to catalog before/after slinky correction
# express in velocity difference
# (1 - wave_measured / wave_catalog) * c
