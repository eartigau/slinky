"""
Batch driver for NIRPS processing

This interactive script orchestrates three processing stages for NIRPS/NIRPS-HE data:
  1) Slinky: updates/patches wavelength solutions and propagates headers, creating
     copies with a "_slinky" suffix.
  2) Pixel PCA (pixpca): computes residuals via pixel-space PCA and stores outputs.
  3) LBL on slinky files: runs line-by-line processing for normal, _slinky and
     _slinky_[batchname] files.

Usage
-----
    python batch_slinky.py <yaml_file>

Required YAML parameters
------------------------
The following keys are expected to be provided by the YAML file (as produced/used
by pixpca.get_params). Only the keys used here are listed; additional keys may
be used by the called modules.

- waveref_file_name: Full path to the reference wavelength solution FITS file.
- template_string:   A format string with one "{}" placeholder for the object name,
                     pointing to the on-disk template FITS files.
- object_of_interest: List of target object names to be processed.
- hotstars:           List/array of telluric (hot) star names for template checks.
- path_to_red:        Glob pattern used to locate reduced files on disk.
- output_slinky:      Root output directory where per-object slinky products live.
- patched_wavesol:    Directory for patched wavelength solutions (used by reset).
- pca_mef_dir:        Directory for Pixel-PCA multi-extension FITS (used by reset).
- residual_path:      Directory for residuals (used by reset).

Workflow
--------
1) Pre-flight checks:
   - Verify the YAML file exists and load parameters via get_params.
   - Verify the wavelength reference file exists.
   - Verify the template file exists for each object_of_interest.
   - Verify reduced files exist for each object_of_interest using path_to_red.
   - For hot stars found in the reduced files, verify their template files exist.

2) Confirmation and options:
   - Ask the user to proceed with batch processing.
   - Optional reset (non-destructive to raw data):
     * Deletes all immediate files under: patched_wavesol, pca_mef_dir, residual_path
       (directories are preserved; deletion is non-recursive).
     * Removes any FITS files inside directories matching "*slinky*" under
       output_slinky/<object>.
     * Errors while deleting individual files are reported and ignored.
   - Ask which stages to run: slinky, pix_pca, lbl_slinky.

3) Execution:
   - Conditionally call:
       slinky.wrap(params)
       pixpca.wrap(params)
       wrap_lbl_slinky.wrapper_slinky(params)

Notes
-----
- This script is conservative and exits upon missing critical inputs.
- It prints human-friendly status with simple checks before running heavy steps.
- The heavy-lifting and detailed behaviors are implemented in the imported modules.
"""

import sys
import slinky
import pixpca
import wrap_lbl_slinky
from residual_pca import run_batch
import os
from pixpca import get_params
import glob

def ask_user(prompt, max_attempts=5):
    """
    Prompt the user for a yes/no/exit answer, with a limited number of attempts.

    Parameters
    ----------
    prompt : str
        The question to display to the user.
    max_attempts : int
        Maximum number of invalid attempts before aborting.

    Returns
    -------
    bool
        True if user answers 'y', False if 'n'. Exits if 'exit' or too many invalid attempts.
    """
    attempts = 0
    print("=" * 80)
    while True:
        answer = input(f"{prompt} (y/n/exit): ").strip().lower()
        if answer == 'exit':
            print("Exiting as requested.")
            sys.exit(0)
        if answer in ('y', 'n'):
            return answer == 'y'
        attempts += 1
        print("\n\tPlease answer with 'y', 'n', or 'exit' : ")
        if attempts >= max_attempts:
            print("Too many invalid attempts. Aborting script.")
            sys.exit(1)

def main():
    """
    Main function for NIRPS batch processing.
    Runs slinky, pixel PCA, and LBL steps for NIRPS data, with user prompts and checks.
    """
    # -------------------------------------------------------------------------
    # Check command-line arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) != 2:
        print("Usage: python batch_nirps.py <yaml_file>")
        sys.exit(1)
    yaml_file = sys.argv[1]

    # -------------------------------------------------------------------------
    # Define options and explanations for user prompts
    # -------------------------------------------------------------------------
    options = {
        'reset': {
            'explanation': """\tDo you want to reset the processing?
\tThis will delete all files in the output directory and start from scratch.
\tUse this if you want to reprocess everything from the beginning.
\t
\tThis is useful if you have made changes to the parameters or the data.
\tThis will not delete the original data, only the processed files.
\tThis is not necessary to run every time, but it is good to do it
\tif you want to start from scratch.
"""        },

        'run_slinky': {
            'explanation': """
\tDo you want to run the slinky step?
\tThis takes a few hours, it reads all the wavelength solutions ever, 
\tdoes again the slinky and updates all headers of all files and makes 
\ta copy with the _slinky suffix.
\tIt is not necessary to run this every time, but it is good to do it
\tto update the wavelength solutions.
"""
        },
        'run_pix_pca': {
            'explanation': """
\tDo you want to run the pixel PCA step?
\tRuns the pixel PCA step.
"""
        },
        'run_lbl_slinky': {
            'explanation': """
\tDo you want to run the LBL on the slinky files?
\tThis runs the LBL on the normal, _slinky and _slinky_[batchname] files.
\tThis takes between 30 min and a few hours per object.
"""
        }
    }

    # -------------------------------------------------------------------------
    # Initialize flags for each processing step
    # -------------------------------------------------------------------------
    do_reset = False
    do_slinky = False
    do_pixpca = False
    do_lbl_slinky = False

    # -------------------------------------------------------------------------
    # Welcome message and terminal width
    # -------------------------------------------------------------------------
    width_terminal = os.get_terminal_size().columns
    print("=" * width_terminal)
    print("Welcome to the NIRPS batch processing script!")
    print("This script will run the slinky, pixel PCA and LBL steps for NIRPS data.")
    print('We first do some checks to see if the data is ready for processing.')

    exit_flag = False

    # -------------------------------------------------------------------------
    # Check that the YAML parameter file exists
    # -------------------------------------------------------------------------
    file_exists = os.path.exists(yaml_file)
    if not file_exists:
        print(f"\t❌\tError: The file {yaml_file} does not exist.")
        exit_flag = True
    else:
        print(f"\t✅\tFile {yaml_file} found. ")

    # -------------------------------------------------------------------------
    # Load parameters from YAML file
    # -------------------------------------------------------------------------
    params = get_params(yaml_file)

    # -------------------------------------------------------------------------
    # Check that the wavelength reference file exists
    # -------------------------------------------------------------------------
    waveref_file_name = params.get('waveref_file_name', None)
    if os.path.exists(waveref_file_name):
        print(f"\t✅\tWavelength reference file {waveref_file_name} found.")
    else:
        print(f"\t❌\tError: Wavelength reference file {waveref_file_name} does not exist.")
        print("\tPlease run the wavelength reference step first.")
        exit_flag = True

    # -------------------------------------------------------------------------
    # Check that templates exist for all host stars
    # -------------------------------------------------------------------------
    for star in params['object_of_interest']:
        template_file = params['template_string'].format(star)
        if os.path.exists(template_file):
            print(f"\t✅\tTemplate file for {star} found: {template_file}")
        else:
            print(f"\t❌\tError: Template file for {star} does not exist: {template_file}")
            print("\tPlease run the template creation step first.")
            exit_flag = True

    # -------------------------------------------------------------------------
    # Check that reduced files exist for all objects of interest
    # -------------------------------------------------------------------------
    files = glob.glob(params['path_to_red'])
    for star in params['object_of_interest']:
        files_obj = [f for f in files if '/'+star+'/' in f]
        if len(files_obj) == 0:
            print(f"\t❌\tError: No reduced files found for {star} in {params['path_to_red']}.")
            print("\tPlease run the reduction step first.")
            exit_flag = True
        print(f"\t✅\tWe found {len(files_obj)} reduced files found for {star} objects in {params['path_to_red']}.")

    # -------------------------------------------------------------------------
    # Check that reduced files and templates exist for all hot stars
    # -------------------------------------------------------------------------
    for hot_stars in params['hotstars']:
        files_obj = [f for f in files if '/'+hot_stars+'/' in f]
        if len(files_obj) == 0:
            continue
        print(f"\t✅\tWe found {len(files_obj)} reduced files found for {hot_stars} objects in {params['path_to_red']}.")
        # check that the hot star template exists
        template_file = params['template_string'].format(hot_stars)
        if os.path.exists(template_file):
            print(f"\t✅\tTemplate file for {hot_stars} found: {template_file}")
        else:
            print(f"\t❌\tError: Template file for {hot_stars} does not exist: {template_file}")

    # -------------------------------------------------------------------------
    # Ask user if they want to proceed with batch processing
    # -------------------------------------------------------------------------
    if not ask_user("Do you want to run the batch processing?"):
        print("Exiting as requested.")
        exit_flag = True

    if exit_flag:
        print("Exiting due to errors or user request.")
        sys.exit(1)
    print("=" * width_terminal)

    # -------------------------------------------------------------------------
    # Ask user which steps to run
    # -------------------------------------------------------------------------

    # Offer an optional full reset of processed outputs before running steps
    # - Paths included: patched wavelength solutions, PCA MEF directory, and residuals
    # - Behavior: delete all files directly under these directories (non-recursive);
    #             directories themselves are preserved
    # - Safety: failures to delete a given file are caught and reported but do not abort
    if ask_user(options['reset']['explanation']):
        # Build the list of output directories to clean
        paths = params['patched_wavesol'], params['pca_mef_dir'], params['residual_path'],params['calib_dir']+'/*slinky*'

        #/space/spirou/SLINKY/calib_NIRPS_HE/

        for path in paths:
            # Skip cleaning for paths that do not exist
            if os.path.exists(path):
                print(f"Deleting all files in {path} to reset processing.")
                # Note: this only deletes immediate children (no subdirectories)
                for file in glob.glob(os.path.join(path, '*')):
                    try:
                        print(f"Deleting file: {file}")
                        os.remove(file)
                    except Exception as e:
                        # Continue on errors, just report them
                        print(f"Error deleting file {file}: {e}")
            else:
                print(f"Path {path} does not exist. No files to delete.")

        # Additionally, remove slinky products per object of interest
        # Pattern matches both "_slinky" and "_slinky_<batchname>" directories
        # and removes any FITS files within them.
        for star in params['object_of_interest']:
            slinky_files = glob.glob(os.path.join(params['output_slinky'], star + '*slinky*', '*.fits'))
            for file in slinky_files:
                try:
                    print(f"Deleting file: {file}")
                    os.remove(file)
                except Exception as e:
                    # Continue on errors, just report them
                    print(f"Error deleting file {file}: {e}")

    # Then prompt which processing steps to run
    if ask_user(options['run_slinky']['explanation']):
        do_slinky = True
    if ask_user(options['run_pix_pca']['explanation']):
        do_pixpca = True
    if ask_user(options['run_lbl_slinky']['explanation']):
        do_lbl_slinky = True
    print("=" * width_terminal)

    # -------------------------------------------------------------------------
    # Run selected processing steps
    # -------------------------------------------------------------------------
    if do_slinky:
        slinky.wrap(params)
    if do_pixpca:
        pixpca.wrap(params)
    if do_lbl_slinky:
        wrap_lbl_slinky.wrapper_slinky(params)

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()