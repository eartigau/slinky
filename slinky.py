import glob
import os
from shutil import copyfile
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.constants import c
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from tqdm import tqdm
from pixtools import doppler, lowpassfilter, sigma, save_pickle, \
    read_pickle, printc, snail, dict2mef, mef2dict, load_yaml, read_t, write_t

from etienne_tools import mjd_to_matplotlib_date, odd_ratio_mean

def val_cheby(coeffs: np.ndarray, xvector: Union[np.ndarray, int, float],
              domain: List[float]) -> Union[np.ndarray, int, float]:
    """
    Using the output of fit_cheby calculate the fit to x  (i.e. y(x))
    where y(x) = T0(x) + T1(x) + ... Tn(x)

    :param coeffs: output from fit_cheby
    :param xvector: x value for the y values with fit
    :param domain: domain to be transformed to -1 -- 1. This is important to
    keep the components orthogonal. For SPIRou orders, the default is 0--4088.
    You *must* use the same domain when getting values with fit_cheby
    :return: corresponding y values to the x inputs
    """
    # transform to a -1 to 1 domain
    domain_cheby = 2 * (xvector - domain[0]) / (domain[1] - domain[0]) - 1
    # fit values using the domain and coefficients
    yvector = np.polynomial.chebyshev.chebval(domain_cheby, coeffs)
    # return y vector
    return yvector

def gp_project(x, y, yerr, wslinky=1e-1, xmin=980, xmax=1850, npts=100000):
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


def odd_ratio_linfit(x, y, yerr):
    """
    Fit a linear model to the data using an iterative weighted least squares method.

    :param x: Abscissa
    :param y: Ordinate
    :param yerr: Error on the ordinate
    :return: Linear fit and error on the fit
    """
    # Remove NaN values
    g = np.isfinite(y + yerr + x)
    x = x[g]
    y = y[g]
    yerr = yerr[g]
    # Initialize weights
    w = np.ones(len(x))

    # Iterate until weights converge
    sum = 1.0
    sum0 = 0.0
    while np.abs(sum0 - sum) > 1e-6:
        sum0 = np.sum(w)
        # Fit the data with current weights
        fit, sig = np.polyfit(x, y, 1, w=w / yerr, cov=True)
        errfit = np.sqrt(np.diag(sig))
        # Compute residuals and update weights
        res = (y - np.polyval(fit, x)) / yerr
        p1 = np.exp(-0.5 * res ** 2)
        p2 = 1e-6
        w = p1 / (p1 + p2)
        sum = np.sum(w)

    return fit, errfit


def sigma(v):
    """
    Compute the standard deviation of the data using a robust estimator.

    :param v: Vector
    :return: Half-width of the 68% confidence interval
    """
    n1, p1 = np.nanpercentile(v, [16, 84])
    return 0.5 * (p1 - n1)

def search_fits_with_mjd(search_string, mjdkey='MJDMID'):
    """
    Search for FITS files with a given search string and sort them by MJD.

    :param search_string: Search string
    :param mjdkey: MJD key in the header
    :return: Files sorted by MJD, MJDs sorted
    """
    files = glob.glob(search_string)
    mjds = np.zeros(len(files))
    for ifile, file in enumerate(files):
        h = fits.getheader(file)
        mjds[ifile] = h[mjdkey]
    order = np.argsort(mjds)
    files = np.array(files)[order]
    mjds = mjds[order]

    return files, mjds

def padding_wavesol(params):
    """
    This function processes FITS files for a given instrument by adding wave solution data.
    It determines the directories for calibration and output based on the user and instrument.
    It then processes each input file, checks for the corresponding wave solution file, 
    and updates the FITS file with the wave solution data.

    Parameters:
    instrument (str): The instrument name (default is 'NIRPS_HE').

    Raises:
    ValueError: If the instrument is not recognized.
    """

    instrument = params['instrument']
    path_to_red = params['path_to_red']
    WAVEFILE_KEY = params['WAVEFILE_KEY']
    fiber = params['fiber']
    

    all_files = np.array(glob.glob(path_to_red))

    objs = np.append(params['object_of_interest'], params['hotstars'])
    keep = np.zeros_like(all_files, dtype=bool)
    for i in tqdm(range(len(all_files)), leave=False):
        for obj in objs:
            if obj in all_files[i]:
                keep[i] = True

    all_files = all_files[keep]

    keep = np.zeros_like(all_files, dtype=bool)
    for i, file in tqdm(enumerate(all_files),leave=False):
        try:
            os.stat(file)
            keep[i] = True
        except:
            printc(f'File {file} not found', 'red')
            continue
    all_files = all_files[keep]

    # Path to wave solution files
    path_to_wavesol = f'{params["patched_wavesol"]}/*.fits'

    # Get all wave solution files and input files
    all_wave_sol_files = np.array(glob.glob(path_to_wavesol))

    # Process each input file
    for ifile, file in enumerate(all_files):
        # Check if the file is a valid FITS file
        printc(f'Processing file {ifile + 1}/{len(all_files)}', 'green')

        try:
            outdir = params['output_slinky'] + file.split('/')[-2] + '/'
            outdir_slinky = params['output_slinky'] + file.split('/')[-2] + '_slinky/'

            if not os.path.isdir(outdir):
                cmd = 'mkdir ' + outdir
                os.system(cmd)
                printc(f'We create the output directory: {outdir}', 'green')

            if not os.path.isdir(outdir_slinky):
                cmd = 'mkdir ' + outdir_slinky
                os.system(cmd)
                printc(f'We create the output directory: {outdir_slinky}', 'green')

            outname = outdir + file.split('/')[-1]
            outname_slinky = outdir_slinky + file.split('/')[-1].replace('.fits', '_slinky.fits')

            printc(f'Processing file: {file}', 'green')

            # Read the header of the input file to get the wave solution file name
            hdr = fits.getheader(file, ext=1)
            wavefile = hdr[WAVEFILE_KEY]

            # Check if the wave solution file exists
            keep = np.array([wavefile in w for w in all_wave_sol_files])
            if True not in keep:
                printc(f'No wave solution found for file: {file}', 'red')
                continue

            if os.path.exists(outdir+file.split('/')[-1]):
                print(outdir)
                print(file)
                # we remove the file if it exists
                printc(f"Removing file: {outdir+file.split('/')[-1]}", 'red')
                os.remove(outdir+file.split('/')[-1])

            # Copy the input ffile to the output directory
            copyfile(file, outdir + file.split('/')[-1])

            if os.path.exists(outname_slinky):
                # we remove the file if it exists
                printc(f'Removing file: {outname_slinky}', 'red')
                os.remove(outname_slinky)

            # Get the wave solution data
            wave_sol_file = all_wave_sol_files[keep][0]
            wavesol = fits.getdata(wave_sol_file)

            # copy outname to outname_slinky
            copyfile(outname, outname_slinky)

            printc(f'Updated file with wave solution: {outname_slinky}', 'green')
            # Update the WaveA extension in the output file with the wave solution data
            hdu = fits.open(outname_slinky, mode='update')
            hdu[f'Wave{fiber}'].data = wavesol
            hdu.close()

        except Exception as e:
            printc(f'Error processing file: {file} - {e}', 'red')
            continue

def refine_wavesol(params):
    # Leverage point of the linear fit, typically the RV content barycenter of the domain
    wave_leverage = 1600
    # Autocorrelation length of the (semi)GP kernel used to model the slinky effect
    wslinky = 1e-1

    doplot = params['doplot']

    calib_dir = params["calib_dir"]
    patched_dir = params["patched_wavesol"]

    if not os.path.isdir(calib_dir):
        printc(f'We create the output directory: {calib_dir}', 'green')
        os.mkdir(calib_dir)

    if not os.path.isdir(patched_dir):
        printc(f'We create the output directory: {patched_dir}', 'green')
        os.mkdir(patched_dir)

    if params["instrument"].upper() == 'SPIROU':
        maestria_path = 'spirou_offline'
        inst_short = 'spirou'
        fiber = 'AB'
        # FOR NIRPS_HE only, probably OK for NIRPS_HA, **not** of for SPIRou
        inst_wavestart = 965
        inst_waveend = 2500
    elif params["instrument"].upper() == 'NIRPS_HE':
        maestria_path = 'nirps_he_online'
        inst_short = 'nirps'
        fiber = 'A'
        # FOR NIRPS_HE only, probably OK for NIRPS_HA, **not** of for SPIRou
        inst_wavestart = 965
        inst_waveend = 1950
    elif params["instrument"].upper() == 'NIRPS_HA':
        maestria_path = 'nirps_ha_online'
        inst_short = 'nirps'
        fiber = 'A'
        # FOR NIRPS_HE only, probably OK for NIRPS_HA, **not** of for SPIRou
        inst_wavestart = 965
        inst_waveend = 1950
    else:
        raise ValueError(f'Instrument {params["instrument"]} not recognized')

    if params['whoami'] == 'eartigau':
        # Sync files from the server to the local calibration directory
        cmd = f'rsync -avz /home/eartigau/scratch/nirps_tempo/calib/*lines_{fiber}.fits' \
                f' {calib_dir}'
        os.system(cmd)
        cmd = f'rsync -avz /home/eartigau/scratch/nirps_tempo/calib/*_wavesol_ref_{fiber}.fits {calib_dir}'
        os.system(cmd)
        cmd = f'rsync -avz /home/eartigau/scratch/nirps_tempo/calib/*pp_e2dsff_{fiber}_wave_night_{fiber}.fits {calib_dir}'
        os.system(cmd)

        plot_folder = '/home/eartigau/scratch/SLINKY/glitch_fp/plots/'

    elif params['whoami'] == 'spirou':
        # Sync files from the server to the local calibration directory
        cmd = f'rsync -avz /cosmos99/{inst_short}/apero-data/{maestria_path}/red/*/*lines_{fiber}.fits' \
                f' {calib_dir}'
        os.system(cmd)
        cmd = f'rsync -avz /cosmos99/{inst_short}/apero-data/{maestria_path}/calib/*_wavesol_ref_{fiber}.fits {calib_dir}'
        os.system(cmd)
        cmd = f'rsync -avz /cosmos99/{inst_short}/apero-data/{maestria_path}/calib/*pp_e2dsff_{fiber}_wave_night_{fiber}.fits {calib_dir}'
        os.system(cmd)

        plot_folder = '/space/spirou/LBL-PCA/wraps/slinky_pca/plots'
        
    # make plot_folder if it does not exist
    if not os.path.isdir(plot_folder):
        printc(f'We create the output directory: {plot_folder}', 'green')
        os.mkdir(plot_folder)

    # find which file is the ref file
    ref_file_hc = glob.glob(f'{calib_dir}/*waveref_hclines*{fiber}.fits')[0]
    tbl_hc_ref = Table.read(ref_file_hc)

    # Load the reference wave solution
    wavesol = glob.glob(f'{calib_dir}/*_wavesol_ref_{fiber}.fits')[0]
    ref_wave_sol = fits.getdata(wavesol)

    # Find all FP and HC files and their corresponding MJDs
    files_fp, mjds_fp = search_fits_with_mjd(f'{calib_dir}/*wave_fplines_{fiber}.fits')
    files_hc, mjds_hc = search_fits_with_mjd(f'{calib_dir}/*wave_hclines_{fiber}.fits')

    dt = np.zeros(len(files_fp))
    for i_fp in range(len(files_fp)):
        dt[i_fp] = np.min(np.abs(mjds_fp[i_fp] - mjds_hc))
    g = dt < 0.1 # HC and FP need to be matched within 0.1 day
    files_fp = files_fp[g]
    mjds_fp = mjds_fp[g]

    # Process each HC file
    for i_hc in range(len(files_hc)):

        file_hc_updated = files_hc[i_hc].replace('.fits', '_slinky.fits')
        if os.path.isfile(file_hc_updated):
            printc(f'File {file_hc_updated} already processed, skipping', 'yellow')
            continue

        printc(f'Processing file {i_hc + 1}/{len(files_hc)}', 'green')
        hdr = fits.getheader(files_hc[i_hc])
        tbl_hc = Table.read(files_hc[i_hc], 'WAVE_HCLIST')

        tbl_hc_tmp = Table(tbl_hc_ref)

        ii = np.zeros(len(tbl_hc_tmp), dtype=int)
        for iline in tqdm(range(len(tbl_hc_tmp)), leave = False):
            g = (tbl_hc_tmp['ORDER'][iline] == tbl_hc['ORDER'])*(tbl_hc_tmp['WAVE_REF'][iline] == tbl_hc['WAVE_REF'])
            if np.sum(g) == 0:
                printc(iline, 'yellow')
                continue
            g = np.where(g)[0][0]
            
            ii[iline] = g

        tbl_hc = Table(tbl_hc[ii])

        if 'CAVITY' in tbl_hc.colnames:
            printc(f'Cavity already computed, skipping', 'yellow')
            continue
        else:
            printc(f'Computing cavity for {files_hc[i_hc]}', 'green')

        # Find the corresponding FP file
        i_fp = np.argmin(np.abs(mjds_fp - mjds_hc[i_hc]))
        delta_t = mjds_fp[i_fp] - mjds_hc[i_hc]

        printc(f'Delta t = {delta_t}', 'green')
        if np.abs(delta_t) > 0.5:
            printc(f'No corresponding FP file found, skipping', 'red')
            continue

        tbl_fp = Table.read(files_fp[i_fp], 'WAVE_FPLIST')

        tbl_hc['CAVITY'] = np.nan
        current_order = -1

        mask = tbl_hc['PIXEL_MEAS'].mask

        all_frac_peak = np.zeros_like(tbl_hc['PIXEL_MEAS'])
        all_cavity = np.zeros_like(tbl_hc['PIXEL_MEAS'])
        wave_ref = np.array(tbl_hc['WAVE_REF'])
        pixel_meas = np.array(tbl_hc['PIXEL_MEAS'])

        for i in tqdm(range(len(tbl_hc)), leave=False):
            if mask[i]:
                continue

            order = tbl_hc['ORDER'][i]
            if order != current_order:
                tbl_fp_order = tbl_fp[tbl_fp['ORDER'] == order]
                tbl_fp_order = tbl_fp_order[~tbl_fp_order['PIXEL_MEAS'].mask]
                current_order = order

                spl = ius(tbl_fp_order['PIXEL_MEAS'], tbl_fp_order['PEAK_NUMBER'], k=1, ext=1)

            all_frac_peak[i] = spl(pixel_meas[i])
            all_cavity[i] = wave_ref[i] * all_frac_peak[i]

        bad = all_frac_peak == 0
        all_frac_peak[bad] = np.nan
        all_cavity[bad] = np.nan

        tbl_hc['PEAK_NUMBER'] = all_frac_peak
        tbl_hc['CAVITY'] = all_cavity

        # Update the FITS file with the new table
        for col in tbl_hc.colnames:
            try:
                tbl_hc[col][tbl_hc[col].mask] = np.nan
            except:
                pass

        copyfile(files_hc[i_hc], file_hc_updated)

        with fits.open(file_hc_updated) as hdul:
            tbl_hc = tbl_hc.as_array()
            hdul[1].data = tbl_hc
            hdul.writeto(file_hc_updated, overwrite=True)

    # Load a reference table and header
    tbl_hc_ref = Table.read(ref_file_hc)
    href = fits.getheader(ref_file_hc)
    cavity_polynomial = np.array([href[key] for key in href['WCAV0*'].keys()])
    WCAV_PED = href['WCAV_PED']

    # Find all wave HC files
    order = np.argsort(mjds_hc)
    mjds_hc = mjds_hc[order]
    files_hc = files_hc[order]

    # Initialize an array to store cavity values
    all_cavity = np.zeros([len(files_hc), len(tbl_hc_ref)], dtype=float)
    for ifile, file in tqdm(enumerate(files_hc), leave=False):
        file_hc_updated = file.replace('.fits', '_slinky.fits')
        all_cavity[ifile] =  Table.read(file_hc_updated, 'WAVE_HCLIST')['CAVITY'].data.data

    # Replace zero values with NaN
    all_cavity[all_cavity == 0] = np.nan
    # Compute the median cavity per line
    med_per_line = np.nanmedian(all_cavity, axis=0)

    meds = np.zeros(len(files_hc))
    # Normalize the cavity values
    for iepoch in range(len(files_hc)):
        meds[iepoch] = np.nanmedian(all_cavity[iepoch] - med_per_line)
        all_cavity[iepoch] -= meds[iepoch]

    #if doplot:
    plt.plot(mjds_hc, meds, '.')
    if not doplot:
        plt.savefig(f'{plot_folder}/cavity_median_{params["instrument"]}.pdf')
    else:
        plt.show()
    plt.close()

    # Update the median cavity per line
    med_per_line = np.nanmedian(all_cavity, axis=0)

    domain = [inst_wavestart, inst_waveend]
    cavity_ref = val_cheby(cavity_polynomial, tbl_hc_ref['WAVE_REF'], domain=domain) + WCAV_PED
    dv_ref = c * (med_per_line / cavity_ref - 1)
    bad = np.abs(dv_ref) > 1000
    dv_ref[bad] = np.nan
    # we offset the reference cavity to match the median cavity
    tbl_hc_ref['WAVE_REF'] = tbl_hc_ref['WAVE_REF'] * (1 - dv_ref / c)

    med_per_line[bad] = np.nan

    n1, p1 = np.nanpercentile(all_cavity, [16, 84], axis=0)
    sig_per_line = (p1 - n1) / 2

    bad = np.sum(np.isfinite(all_cavity), axis=0) < all_cavity.shape[0] // 5
    sig_per_line[sig_per_line == 0] = np.nan
    sig_per_line[bad] = np.nan

    sig_per_line_ms = c * (sig_per_line / med_per_line)

    # Initialize arrays to store fit parameters
    all_slopes = np.zeros_like(files_hc, dtype=float)
    all_errslopes = np.zeros_like(files_hc, dtype=float)
    all_pedestals = np.zeros_like(files_hc, dtype=float)
    all_errpedestals = np.zeros_like(files_hc, dtype=float)

    # Create a figure for plotting with 2 rows and 1 column, sharing the x-axis
    fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
    # Initialize the plots with NaN values to set up the date format
    ax[0].plot_date([np.nan], [np.nan])
    ax[1].plot_date([np.nan], [np.nan])

    # Loop through each wave HC file
    for ifile, file in enumerate(files_hc):
        file_hc_updated = file.replace('.fits', '_slinky.fits')
        # Read the table from the current file
        tbl2 = Table.read(file_hc_updated, 'WAVE_HCLIST')

        # Get the header of the current file
        h = fits.getheader(file_hc_updated)

        # Calculate the cavity deviation from the median per line, normalized by WCAV_PED and converted to m/s
        dcavity = c * (tbl2['CAVITY'].data.data / med_per_line - 1)  # / WCAV_PED
        sdcavity = c * sig_per_line / med_per_line
        # Get the NSIG values from the table
        nsig = tbl2['NSIG'].data.data
        # Normalize the WAVE_REF values and shift by the leverage point
        wave2 = tbl2['WAVE_REF'] / 1e3 - wave_leverage / 1e3
        # Initialize the fit parameters
        fit = [1, 0]
        sig_fit = [1, 1]

        # fit[0],sig_fit[0] = odd_ratio_mean(dcavity, sdcavity)
        moy, err = odd_ratio_mean(dcavity, sdcavity)
        fit[1] = moy
        sig_fit[1] = err
        slope, err_slope = odd_ratio_mean((dcavity - moy) / wave2, sdcavity / wave2)
        fit[0] = slope
        sig_fit[0] = err_slope

        # Perform a weighted linear fit to the data
        fit, sig_fit = odd_ratio_linfit(wave2, dcavity - moy, sdcavity)
        fit[1] += moy

        # Print the fit results for the current file
        args = (ifile + 1, len(files_hc), fit[1], sig_fit[1], fit[0], sig_fit[0])
        printc(f'\t{args[0]}/{args[1]} zp {args[2]:5.2f}+-{args[3]:5.2f} [m/s], slope {args[4]:5.2f}+-{args[5]:5.2f} [m/s/µm]', 'green')

        if doplot:
            # Plot the zero-point and slope with error bars
            ax[0].errorbar(mjd_to_matplotlib_date(h['MJDMID']), fit[1], yerr=sig_fit[1], fmt='.g')

            ax[1].errorbar(mjd_to_matplotlib_date(h['MJDMID']), fit[0], yerr=sig_fit[0], fmt='.g')

        # Store the fit parameters and their errors
        all_slopes[ifile] = fit[0]
        all_errslopes[ifile] = sig_fit[0]
        all_pedestals[ifile] = fit[1]
        all_errpedestals[ifile] = sig_fit[1]
        # Store the MJD of the current file

    ax[0].grid(color='grey', linestyle='--', linewidth=0.5)
    ax[1].grid(color='grey', linestyle='--', linewidth=0.5)
    ax[0].set(ylabel='Zero-point [m/s]')
    ax[1].set(ylabel='Slope [m/s/µm]', xlabel='MJD')
    ax[1].set(xlabel='Date')
    # Display the plot
    if doplot:
        plt.show()
    else:
        plt.savefig(f'{plot_folder}/wavesol_{params["instrument"]}.png')
        plt.close()

    # Print the standard deviation of the differences between consecutive pedestals
    printc(f'Standard deviation of differences between consecutive pedestals: {sigma(all_pedestals - np.roll(all_pedestals, 1)) / np.sqrt(2)}', 'green')

    recovered_pedestal = np.zeros_like(all_pedestals)
    recovered_slope = np.zeros_like(all_slopes)
    recovered_errslope = np.zeros_like(all_errslopes)
    recovered_errpedestal = np.zeros_like(all_errpedestals)

    # Loop through each FP file
    for i_fp in range(len(files_fp)):
        printc('\n', 'green')
        # get widht of the terminal window
        n_pix_terminal = os.get_terminal_size().columns

        printc('*' * n_pix_terminal, 'green')
        printc('\n', 'green')
        printc(f'Processing file {i_fp + 1}/{len(files_fp)}', 'green')
        printc('\n', 'green')

        printc(f'Processing wavelength solution of night {i_fp + 1}/{len(files_fp)}', 'green')
        file_fp = files_fp[i_fp]
        hdr = fits.getheader(file_fp)
        wavefile = f'{calib_dir}' + hdr['WAVEFILE']
        patched_wavefile = f'{patched_dir}' + hdr['WAVEFILE']


        hdr = fits.getheader(wavefile)

        tbl_fp = Table.read(file_fp, 'WAVE_FPLIST')
        hdr_fp = fits.getheader(file_fp)

        # Find the corresponding HC (Hollow Cathode) file based on the closest MJD
        i_hc = np.argmin(np.abs(mjds_hc - hdr_fp['MJDMID']))
        slope_hc = all_slopes[i_hc]
        err_slope_hc = all_errslopes[i_hc]
        pedestal_hc = all_pedestals[i_hc]
        err_pedestal_hc = all_errpedestals[i_hc]

        if os.path.isfile(patched_wavefile):
            printc(f'File already processed, skipping', 'yellow')
            continue


        # Apply the polynomial cavity correction to the wavelength model
        wavelength_model = np.array(tbl_fp['WAVE_REF'].data)
        doppler = (wavelength_model / 1e3 - wave_leverage / 1e3) * slope_hc + pedestal_hc
        cavity = val_cheby(cavity_polynomial, wavelength_model, domain=[inst_wavestart, inst_waveend]) + WCAV_PED
        peak_number = tbl_fp['PEAK_NUMBER'].data
        wavelength_model = cavity / peak_number * (1 + doppler / c)
        
        tbl_fp['WAVE_REF'] = wavelength_model

        # Initialize the wavelength map with the reference wave solution
        wavemap = np.array(ref_wave_sol)

        dv_residuals = []
        dv_residuals_err = []
        wave_ref_residuals = []

        # Loop through each order in the table
        for order in np.unique(tbl_fp['ORDER']):
            g = tbl_fp['ORDER'] == order
            tbl_order = tbl_fp[g]
            pixel_meas = np.array(tbl_order['PIXEL_MEAS'])
            wave_ref = np.array(tbl_order['WAVE_REF'])
            valid = np.isfinite(pixel_meas + wave_ref)
            pixel_meas = pixel_meas[valid]
            wave_ref = wave_ref[valid]

            # Fit a polynomial to the pixel measurements and wave references
            fit = np.polyfit(pixel_meas, wave_ref, 5)
            residual = (wave_ref / np.polyval(fit, pixel_meas) - 1) * c
            wave_order = np.polyval(fit, np.arange(4088))

            # Update the wavelength map if the wave order is valid
            if np.all(np.isfinite(wave_order)):
                wavemap[order] = np.polyval(fit, np.arange(4088))

            # Calculate the error on the residuals
            err = sigma(residual - np.roll(residual, 1)) / np.sqrt(2)
            err = np.ones_like(residual) * err

            dv_residuals.append(residual)
            dv_residuals_err.append(err)
            wave_ref_residuals.append(wave_ref)

        # Concatenate the residuals and errors
        dv_residuals = np.concatenate(dv_residuals)
        dv_residuals_err = np.concatenate(dv_residuals_err)
        wave_ref_residuals = np.concatenate(wave_ref_residuals)

        # We have 5 points per scale length of slinky
        npts = int((np.nanmax(wavemap) - np.nanmin(wavemap)) / wslinky * 5)
        xmin = np.nanmin(wavemap)
        xmax = np.nanmax(wavemap)

        # Project the residuals onto a regular grid using a Gaussian weight
        xv, yv = gp_project(wave_ref_residuals, dv_residuals, dv_residuals_err, wslinky=wslinky, xmin=xmin, xmax=xmax,
                            npts=npts)
        spl = ius(xv, yv, k=2)

        # Correct the wavelength map for the cavity effect
        wavemap = wavemap * (1 + spl(wavemap) / c)

        # this is a sanity check that the wavelength solution is now consistent. We backproject the HC line position
        # onto the wavelength grid and compare with the nominal HC line position

        i_hc = np.argmin(np.abs(hdr_fp['MJDMID'] - mjds_hc))

        file_hc_updated = files_hc[i_hc].replace('.fits', '_slinky.fits')
        tbl_hc = Table.read(file_hc_updated)
        tbl_hc['WAVE_REF'] = tbl_hc_ref['WAVE_REF']

        dv_fp = []
        wave_fp = []
        dv_hc = []
        wave_hc = []

        hc_meas = np.array(tbl_hc['PIXEL_MEAS'])

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

        for iord in np.unique(tbl_hc['ORDER']):
            g = tbl_hc['ORDER'] == iord
            wave_ord = wavemap[iord]
            pix = np.arange(4088)
            spl = ius(pix, wave_ord, k=2)

            hc_meas[g] = spl(tbl_hc['PIXEL_MEAS'][g])
            v = np.array(tbl_hc['WAVE_REF'][g] / spl(tbl_hc['PIXEL_MEAS'][g]) - 1) * c
            dv_hc.append(v)
            wave_hc.append(tbl_hc['WAVE_REF'][g])

            if doplot:
                ax[0].plot(tbl_hc['WAVE_REF'][g], v, 'o', alpha=0.1)

            g = tbl_fp['ORDER'] == iord

            v = np.array(tbl_fp['WAVE_REF'][g] / spl(tbl_fp['PIXEL_MEAS'][g]) - 1) * c

            if doplot:
                ax[1].plot(tbl_fp['WAVE_REF'][g], v, 'o', alpha=0.1)
            dv_fp.append(v)
            wave_fp.append(tbl_fp['WAVE_REF'][g])



        dv_hc_plot = np.concatenate(dv_hc)
        dv_fp_plot = np.concatenate(dv_fp)
        wave_hc_plot = np.concatenate(wave_hc)
        wave_fp_plot = np.concatenate(wave_fp)

        printc(f'RMS {sigma(dv_fp_plot):.2f} m/s of FP peaks', 'green')


        sig_per_line_ms_plot = np.array(sig_per_line_ms)

        ord_hc = np.argsort(wave_hc_plot)
        ord_fp = np.argsort(wave_fp_plot)

        dv_hc_plot = dv_hc_plot[ord_hc]
        dv_fp_plot = dv_fp_plot[ord_fp]
        wave_hc_plot = wave_hc_plot[ord_hc]
        wave_fp_plot = wave_fp_plot[ord_fp]
        sig_per_line_ms_plot = sig_per_line_ms_plot[ord_hc]

        mean_hcs = []
        err_hcs = []
        mean_waves = []
        for i in range(len(dv_hc_plot)//100):
            mean_hc, sig_hc = odd_ratio_mean(dv_hc_plot[i*100:(i+1)*100], sig_per_line_ms_plot[i*100:(i+1)*100])
            mean_wave_hc = np.mean(wave_hc_plot[i*100:(i+1)*100])
            ax[0].errorbar(mean_wave_hc, mean_hc, yerr=sig_hc, fmt='k.')
            err_hcs.append(sig_hc)  
            mean_hcs.append(mean_hc)
            mean_waves.append(mean_wave_hc)
        for i in range(len(dv_fp_plot)//100):
            mean_fp, sig_fp = np.nanmedian(dv_fp_plot[i*100:(i+1)*100]), sigma(dv_fp_plot[i*100:(i+1)*100])/10.0
            mean_wave_fp = np.nanmean(wave_fp_plot[i*100:(i+1)*100])
            ax[1].errorbar(mean_wave_fp, mean_fp, yerr=sig_fp, fmt='k.')
        
        mean_hcs = np.array(mean_hcs)
        err_hcs = np.array(err_hcs)
        mean_waves = np.array(mean_waves)
        valid = np.isfinite(mean_hcs)*np.isfinite(err_hcs)*np.isfinite(mean_waves)
        mean_hcs = mean_hcs[valid]
        err_hcs = err_hcs[valid]
        mean_waves = mean_waves[valid]

        printc(f'Fitting polynomial to {len(mean_waves)} points', 'green')
        fit,cov = np.polyfit((mean_waves-wave_leverage)/1000, mean_hcs, 1, w=1/err_hcs, cov=True)
        sig = np.sqrt(np.diag(cov))

        printc(f'Slope: {fit[0]:5.2f}+-{sig[0]:5.2f} [m/s/µm]', 'green')
        printc(f'Pedestal: {fit[1]:5.2f}+-{sig[1]:5.2f} [m/s]', 'green')

        all_slopes[i_fp] += fit[0]
        all_pedestals[i_fp] += fit[1]


        # ax[0].plot(tbl_hc['WAVE_REF'],c*(tbl_hc['WAVE_MEAS']/tbl_hc['WAVE_REF']-1),'.')
        ax[0].set(xlabel='Nominal HC line position [nm]')
        ax[1].set(xlabel='Nominal FP line position [nm]')
        ax[0].set(ylabel='RV [m/s]')
        ax[1].set(ylabel='RV [m/s]')

        ax[0].set(ylim=[-30, 30])
        ax[1].set(ylim=[-30, 30])
        ax[0].grid(color='grey', linestyle='--', linewidth=0.5)
        ax[1].grid(color='grey', linestyle='--', linewidth=0.5)
        if doplot:
            plt.show()
        else:
            plt.savefig(f'{plot_folder}/fp_hc_{params["instrument"]}_{i_fp}.png')
            plt.close()

        dv_hc = np.array((hc_meas / tbl_hc['WAVE_REF'] - 1) * c)
        dv_fp = np.concatenate(dv_fp)

        valid = np.isfinite(dv_hc)
        fit, sig = odd_ratio_linfit((tbl_hc['WAVE_REF'] - wave_leverage)[valid] / 1e3, dv_hc[valid],
                                    sig_per_line_ms[valid])

        recovered_pedestal[i_fp] = fit[1]
        recovered_slope[i_fp] = fit[0]
        recovered_errslope[i_fp] = sig[0]
        recovered_errpedestal[i_fp] = sig[1]


        printc(f'Input pedestal: {pedestal_hc:5.2f}+-{err_pedestal_hc:5.2f} [m/s]', 'green')
        printc(f'Input slope: {slope_hc:5.2f}+-{err_slope_hc:5.2f} [m/s/µm]', 'green')
        printc(f'{i_fp} -> zp {fit[1]:5.2f}+-{sig[1]:5.2f} [m/s], slope {fit[0]:5.2f}+-{sig[0]:5.2f} [m/s/µm]', 'green')


        printc(f'RMS HC: {sigma(dv_hc):.2f} m/s', 'green')
        printc(f'RMS FP: {sigma(dv_fp):.2f} m/s', 'green')


        printc(f'Patching {wavefile} and outputting to {patched_wavefile}', 'green')
        # Update the header of the patched wavefile
        # Update the wavefile with the corrected wavelength map
        copyfile(wavefile, patched_wavefile)
        with fits.open(patched_wavefile) as hdul:
            printc(f'Updating file {wavefile}', 'green')
            hdul[1].data = wavemap
            hdul[0].header['SLINKY'] = True, 'Wavelength solution corrected for cavity effect'
            hdul[0].header['ZPCAV'] = pedestal_hc, 'Zero-point [m/s]'
            hdul[0].header['ZPCAVER'] = err_pedestal_hc, 'Error on zero-point [m/s]'
            hdul[0].header['SLPCAV'] = slope_hc, 'Slope [m/s/um]'
            hdul[0].header['SLPCAVER'] = err_slope_hc, 'Error on slope [m/s/um]'
            hdul.writeto(patched_wavefile, overwrite=True)

    printc(f'RMS : {np.std(recovered_pedestal):.3f} m/s pedestal', 'green')
    printc(f'RMS : {np.std(recovered_slope):.3f} m/s/µm slope', 'green')

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].errorbar(mjds_hc, recovered_pedestal, yerr=recovered_errpedestal, fmt='g.')
    ax[1].errorbar(mjds_hc, recovered_slope, yerr=recovered_errslope, fmt='g.')
    ax[0].set(ylabel='Zero-point [m/s]')
    ax[1].set(ylabel='Slope [m/s/µm]', xlabel='MJD')
    ax[1].set(xlabel='Date')

    if doplot:
        plt.show()
    else:
        plt.savefig(f'{params["output_slinky"]}wavesol_{params["instrument"]}_patched.png')
        plt.close()



def wrap(params):
    """
    Run the full slinky wavelength solution refinement and padding process.

    :param instrument: instrument name (default 'NIRPS')
    """
    refine_wavesol(params=params)
    padding_wavesol(params=params)