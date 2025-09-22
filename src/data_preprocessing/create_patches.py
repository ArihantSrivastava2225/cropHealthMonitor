"""
Task 3: Reads individual band files from disk and incrementally builds
the final 6-channel patch array, saving it in .mat format.
"""
import os
import numpy as np
import scipy.io
from PIL import Image, ImageDraw
# --- CRITICAL FIX: Use an absolute import from the src root ---
from data_preprocessing.utils import print_raster_stats

def create_and_save_patches(temp_band_paths, date_str, patch_size, output_mat_path, output_viz_dir):
    """
    Builds the final patch array by reading one band at a time from disk.
    """
    if not temp_band_paths:
        print("      -> No temporary band files found. Skipping patch creation.")
        return

    print(f"    Slicing {date_str} data into patches from temporary files...")
    
    # Load the first band to get dimensions
    try:
        first_band = np.load(temp_band_paths[0])
    except FileNotFoundError:
        print(f"      ERROR: Could not find temporary band file: {temp_band_paths[0]}")
        return
        
    height, width = first_band.shape
    num_channels = len(temp_band_paths)

    patches_y = height // patch_size
    patches_x = width // patch_size
    num_patches = patches_y * patches_x

    if num_patches == 0:
        print("      -> No full patches could be extracted. Skipping file save.")
        return

    # --- Pre-allocate the final 6-channel patch array ---
    print(f"      -> Pre-allocating memory for {num_patches} patches...")
    try:
        patches_array = np.zeros((num_patches, patch_size, patch_size, num_channels), dtype=np.float32)
    except MemoryError:
        print("      ERROR: Not enough RAM to pre-allocate memory for all patches.")
        return
    
    # --- Loop through each band, load it, patch it, and place it in the final array ---
    for c, band_path in enumerate(temp_band_paths):
        print(f"        - Processing channel {c+1}/{num_channels}...")
        band_data = np.load(band_path)
        patch_index = 0
        for y in range(patches_y):
            for x in range(patches_x):
                start_y = y * patch_size
                start_x = x * patch_size
                patch = band_data[start_y : start_y + patch_size, start_x : start_x + patch_size]
                patches_array[patch_index, :, :, c] = patch
                patch_index += 1
        # Free memory from the large band file
        del band_data

    # --- Print Final Data Stats ---
    print_raster_stats(patches_array, f"{date_str} Final Patches")
    
    # --- Save as a .mat file ---
    mat_dict = {'patches': patches_array}
    scipy.io.savemat(output_mat_path, mat_dict)
    print(f"      -> Saved {num_patches} patches to {os.path.basename(output_mat_path)}")

    # --- Create Grid Visualization ---
    base_image_path = os.path.join(output_viz_dir, f"{date_str}_02_after_mask.png")
    if os.path.exists(base_image_path):
        img = Image.open(base_image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        for y in range(patches_y):
            for x in range(patches_x):
                start_y = y * patch_size
                start_x = x * patch_size
                draw.rectangle([start_x, start_y, start_x + patch_size - 1, start_y + patch_size - 1], outline="cyan", width=2)
        
        viz_path = os.path.join(output_viz_dir, f"{date_str}_03_patch_grid.png")
        img.save(viz_path)
        print(f"      -> Patch grid visualization saved.")

