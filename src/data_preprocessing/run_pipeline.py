"""
Main Orchestrator for the Kaggle Preprocessing Pipeline.
This version first creates a writable, converted copy of the dataset
to bypass driver and file system issues on Kaggle.
"""
import os
import shutil
import sys
from collections import defaultdict

# --- CRITICAL FIX: Make the project structure import-aware ---
# This adds the parent 'src' directory to the Python path, allowing absolute imports.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SRC_DIR))

# --- Use absolute imports from the 'src' level ---
from data_preprocessing.config import KAGGLE_INPUT_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EVENT_METADATA, PATCH_SIZE, TARGET_RESOLUTION
from data_preprocessing.jp2_to_tig import prepare_writable_dataset
from data_preprocessing.grid_and_mask import define_event_grid_and_mask
from data_preprocessing.process_and_mosaic import process_and_mosaic_daily_data
from data_preprocessing.create_patches import create_and_save_patches

if __name__ == '__main__':
    # --- STAGE 0: Create a writable, converted copy of the entire dataset ---
    # This solves both the read-only file system and the .jp2 driver issues.
    prepare_writable_dataset(KAGGLE_INPUT_DIR, RAW_DATA_DIR)
    
    # --- The rest of the pipeline now runs on the new writable data ---
    TEMP_DIR = os.path.join(PROCESSED_DATA_DIR, 'temp_bands')
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    for event_name in EVENT_METADATA.keys():
        print(f"\n{'='*20} Processing Event: {event_name} {'='*20}")
        event_raw_dir = os.path.join(RAW_DATA_DIR, event_name)
        event_processed_dir = os.path.join(PROCESSED_DATA_DIR, event_name)
        viz_dir = os.path.join(event_processed_dir, 'visualizations')
        os.makedirs(event_processed_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        if not os.path.isdir(event_raw_dir): 
            print(f"  -> Writable raw data directory not found for {event_name}. Skipping.")
            continue
        
        common_grid, cropland_mask = define_event_grid_and_mask(event_raw_dir, viz_dir, event_name, TARGET_RESOLUTION)
        if common_grid is None: continue

        products_by_date = defaultdict(list)
        date_folders = [d for d in os.listdir(event_raw_dir) if os.path.isdir(os.path.join(event_raw_dir, d))]
        for date_folder in date_folders:
            date_folder_path = os.path.join(event_raw_dir, date_folder)
            product_folders = [os.path.join(date_folder_path, pf) for pf in os.listdir(date_folder_path)]
            products_by_date[date_folder].extend(product_folders)
            
        sorted_dates = sorted(products_by_date.keys())
        for i, date_str in enumerate(sorted_dates):
            print(f"\n  Processing timestep {i+1}/{len(sorted_dates)} (Date: {date_str})...")
            # Filter for .SAFE and Landsat folders, as the converted .tif files are inside them
            product_paths = [p for p in products_by_date[date_str] if 'SAFE' in os.path.basename(p) or 'LC0' in os.path.basename(p)]
            
            temp_band_paths = process_and_mosaic_daily_data(product_paths, common_grid, cropland_mask, viz_dir, date_str, TEMP_DIR)
            
            if temp_band_paths:
                mat_filename = f"patches_{date_str}_{i:02d}.mat"
                mat_path = os.path.join(event_processed_dir, mat_filename)
                create_and_save_patches(temp_band_paths, date_str, PATCH_SIZE, mat_path, viz_dir)

    print("\nCleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)
    print(f"\n{'='*20} PREPROCESSING COMPLETE {'='*20}")
    print(f"Intermediate 6-channel .mat files saved to: {PROCESSED_DATA_DIR}")
    print("Download the 'processed' folder from Kaggle's output to run the local MATLAB script.")

