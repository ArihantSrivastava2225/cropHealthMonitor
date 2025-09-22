"""
Central Configuration File for the Preprocessing Pipeline
*** THIS VERSION IS CONFIGURED FOR THE KAGGLE NOTEBOOK ENVIRONMENT ***
"""
import os

# --- Base Paths for Kaggle ---
# The original, read-only input data directory.
KAGGLE_INPUT_DIR = '/kaggle/input/crophealthsatellitedatav1/organized_date'

# The writable directory inside the Kaggle environment.
KAGGLE_WORKING_DIR = '/kaggle/working'

# The pipeline will create a new, writable copy of the data here.
# ALL subsequent processing will read from this path.
RAW_DATA_DIR = os.path.join(KAGGLE_WORKING_DIR, 'writable_raw_data')

# The final processed output will be saved here.
PROCESSED_DATA_DIR = os.path.join(KAGGLE_WORKING_DIR, 'processed')

# --- Processing Settings ---
PATCH_SIZE = 256
TARGET_RESOLUTION = 30  # meters

# --- Metadata (must match folder names in your input data) ---
EVENT_METADATA = {
    'Bathinda-PinkBollworm': {},
    'EasternUP-RedRot': {},
    'Haryana-RiceBlast': {},
    'Punjab-leafhopper': {},
    'Ropar-wheatRust': {},
    'Una-yellowRust': {}
}

