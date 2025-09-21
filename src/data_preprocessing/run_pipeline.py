"""
Main Orchestrator for the Local Preprocessing Pipeline.

Run this script from the 'src' directory to process all raw data. It will
sequentially execute the defined tasks for each event:
1. Define a universal grid and fetch a master cropland mask.
2. For each day, process, mosaic, and mask all satellite images.
3. Slice the final daily image into patches for the deep learning model.
"""

import os
from collections import defaultdict
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EVENT_METADATA, PATCH_SIZE, TARGET_RESOLUTION
from grid_and_mask import define_event_grid_and_mask
from process_and_mosaic import process_and_mosaic_daily_data
from create_patches import create_and_save_patches

if __name__ == '__main__':
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
        date_folders = [d for d in os.listdir(event_raw_dir) if os.path.isdir(os.path.join(event_raw_dir, d))]
        for date_folder in date_folders:
            date_folder_path = os.path.join(event_raw_dir, date_folder)
            product_folders = [os.path.join(date_folder_path, pf) for pf in os.listdir(date_folder_path)]
            products_by_date[date_folder].extend(product_folders)
            
        # --- Loop through each date and process its data ---
        sorted_dates = sorted(products_by_date.keys())
        for i, date_str in enumerate(sorted_dates):
            print(f"\n  Processing timestep {i+1}/{len(sorted_dates)} (Date: {date_str})...")
            product_paths = products_by_date[date_str]
            
            # --- Task 2: Process, Mosaic, and Mask ---
            masked_daily_data = process_and_mosaic_daily_data(product_paths, common_grid, cropland_mask, viz_dir, date_str)
            
            # --- Task 3: Create Patches ---
            if masked_daily_data is not None:
                # CRITICAL FIX: Change filename extension from .npy to .mat
                mat_filename = f"patches_{date_str}_{i:02d}.mat"
                mat_path = os.path.join(event_processed_dir, mat_filename)
                create_and_save_patches(masked_daily_data, date_str, PATCH_SIZE, mat_path, viz_dir)

    print(f"\n{'='*20} PREPROCESSING COMPLETE {'='*20}")
    print(f"All processed data has been saved to: {os.path.abspath(PROCESSED_DATA_DIR)}")
    print("Check the 'visualizations' subfolder in each processed event directory.")
    print("You can now upload the 'processed' folder (containing .mat files) for your MATLAB workflow.")

