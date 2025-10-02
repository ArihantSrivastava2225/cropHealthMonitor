"""
Central Configuration File for the Preprocessing Pipeline
*** THIS VERSION IS CORRECTED TO MATCH YOUR FOLDER STRUCTURE ***
"""
import os

# --- Base Paths for your Local Machine ---
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'organize', 'organized_date')
# PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
BASE_DIR = "/content/drive/MyDrive/AgriTechPro"
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'organize', 'organized_date')
PROCESSED_DATA_DIR = os.path.join("/content/cropHealthMonitor", 'data', 'processed')

# --- Processing Settings ---
PATCH_SIZE = 256
TARGET_RESOLUTION = 30  # meters

# --- Metadata (must match folder names in your input data) ---
EVENT_METADATA = {
    # 'Bathinda-PinkBollworm': {},
    'EasternUP-RedRot': {},
    # CORRECTED: Replaced 'Maharashtra-Blacksmut' with the actual folder name
    'Haryana-RiceBlast': {},
    # CORRECTED: Matched the case for 'leafHopper'
    'Punjab-leafHopper': {}, 
    'Ropar-wheatRust': {},
    'Una-yellowRust': {}
}

