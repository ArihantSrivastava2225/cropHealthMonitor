"""
Main Orchestrator for the Local Preprocessing Pipeline.
Manages a temporary directory for out-of-core processing of large rasters.
"""
import os
import shutil
import sys
from collections import defaultdict

# --- Make the project structure import-aware ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SRC_DIR))

from data_preprocessing.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EVENT_METADATA, PATCH_SIZE, TARGET_RESOLUTION
from data_preprocessing.grid_and_mask import define_event_grid_and_mask
from data_preprocessing.process_and_mosaic import process_and_mosaic_daily_data
from data_preprocessing.create_patches import create_and_save_individual_patches

if __name__ == '__main__':
    # A temporary directory for storing intermediate band files to save RAM
    TEMP_DIR = os.path.join(PROCESSED_DATA_DIR, 'temp_bands')
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR) # Clean up from any previous runs
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
            print(f"  Raw data directory not found for {event_name}. Skipping.")
            continue
        
        # --- Task 1: Define Universal Grid and Mask for the entire event ---
        common_grid, cropland_mask = define_event_grid_and_mask(event_raw_dir, viz_dir, event_name, TARGET_RESOLUTION)
        if common_grid is None:
            print(f"  Could not define grid for {event_name}. Skipping event.")
            continue

        # --- Group all products by date ---
        products_by_date = defaultdict(list)
        date_folders = sorted([d for d in os.listdir(event_raw_dir) if os.path.isdir(os.path.join(event_raw_dir, d))])
        for date_folder in date_folders:
            date_folder_path = os.path.join(event_raw_dir, date_folder)
            product_folders = [os.path.join(date_folder_path, pf) for pf in os.listdir(date_folder_path)]
            products_by_date[date_folder].extend(product_folders)
            
        # --- Loop through each date and process its data ---
        for i, date_str in enumerate(date_folders):
            print(f"\n  Processing timestep {i+1}/{len(date_folders)} (Date: {date_str})...")
            product_paths = [p for p in products_by_date[date_str] if 'SAFE' in p or 'LC0' in p]
            
            # --- Task 2: Process, Mosaic, and Mask ---
            # This function returns a list of paths to temporary band files.
            temp_band_paths = process_and_mosaic_daily_data(product_paths, common_grid, cropland_mask, viz_dir, date_str, TEMP_DIR)
            
            # --- Task 3: Create Individual Patches ---
            if temp_band_paths:
                # Create a new subdirectory for this specific date to hold all its patch files
                output_dir_for_patches = os.path.join(event_processed_dir, date_str)
                os.makedirs(output_dir_for_patches, exist_ok=True)
                
                create_and_save_individual_patches(temp_band_paths, date_str, PATCH_SIZE, output_dir_for_patches, viz_dir)

    # --- Final Cleanup ---
    print("\nCleaning up temporary files...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    print(f"\n{'='*20} PREPROCESSING COMPLETE {'='*20}")
    print(f"Final data saved as individual .mat patch files in: {os.path.abspath(PROCESSED_DATA_DIR)}")
