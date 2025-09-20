"""
Global Configuration File

This file contains all static configuration for the project, including
file paths, hyperparameters, and metadata.
"""
import os

# --- Base Paths ---
# Assumes 'src' and 'data' are in the same root folder (e.g., 'crop_health_project')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DATA_DIR = os.path.join(BASE_DIR, 'data', 'matlab_enhanced')
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

# --- Model Hyperparameters ---
BATCH_SIZE = 16
N_STEPS_IN = 8      # Number of past satellite images to use as input
N_STEPS_OUT = 5     # Number of future steps to predict
EPOCHS = 20
LEARNING_RATE = 1e-5
FEATURE_VECTOR_SIZE = 128 # Output size of the CNN feature extractor
HEALTH_LOSS_WEIGHT = 0.5  # Weight for the health index prediction loss

# --- Metadata (must match folder names in your matlab_enhanced data) ---
EVENT_METADATA = {
    'Bathinda-PinkBollworm': {'crop_type': 'Cotton', 'disease': 'Bollworm', 'label': 0},
    'EasternUP-RedRot': {'crop_type': 'Sugarcane', 'disease': 'RedRot', 'label': 1},
    'Maharashtra-Blacksmut': {'crop_type': 'Wheat', 'disease': 'Smut', 'label': 2},
    'Punjab-leafhopper': {'crop_type': 'Cotton', 'disease': 'Leafhopper', 'label': 3},
    'Ropar-wheatRust': {'crop_type': 'Wheat', 'disease': 'Rust', 'label': 4},
    'Una-yellowRust': {'crop_type': 'Wheat', 'disease': 'Rust', 'label': 4} # Same label as Rust
}
