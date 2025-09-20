"""
Central Configuration File for the Preprocessing Pipeline

This file contains all the user-configurable settings, such as file paths,
processing parameters, and metadata that drives the pipeline.
This version is updated to match your specific folder structure.
"""
import os

# ==============================================================================
# PART 1: FILE PATHS
# ==============================================================================
# --- Base Paths ---
# This line automatically finds the project's root directory (e.g., 'SERVER').
# It assumes your 'src' folder and 'data' folder are at the same level.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- Input/Output Directories ---
# This path now correctly points to your date-organized satellite data.
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'organize', 'organized_date')

# This is where the final, clean .npy files will be saved.
# A new 'processed' folder will be created inside the 'data' directory.
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


# ==============================================================================
# PART 2: PROCESSING SETTINGS
# ==============================================================================
# --- Image Processing Parameters ---
PATCH_SIZE = 256
TARGET_RESOLUTION = 30  # The target resolution in meters (e.g., 30m for Landsat)


# ==============================================================================
# PART 3: PROJECT METADATA
# ==============================================================================
# --- Event Metadata ---
# The keys in this dictionary MUST EXACTLY MATCH the names of your event
# folders inside the 'organized_date' directory, as shown in your screenshot.
EVENT_METADATA = {
    # 'Bathinda-PinkBollworm': {},
    'EasternUP-RedRot': {},
    'Haryana-RiceBlast': {},
    'Punjab-leafhopper': {},
    'Ropar-wheatRust': {},
    'Una-yellowRust': {}
}

