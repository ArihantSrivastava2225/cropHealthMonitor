"""
Task 3: Slices a large, masked data array into smaller patches for the CNN.
This version saves the final output in .mat format for MATLAB compatibility
and uses a memory-efficient pre-allocation strategy.
"""
import os
import numpy as np
import scipy.io  # Import SciPy for saving .mat files
from PIL import Image, ImageDraw
from utils import print_raster_stats

def create_and_save_patches(masked_data_array, date_str, patch_size, output_mat_path, output_viz_dir):
    """
    Slices a data array into patches, saves them as a .mat file,
    and creates a visualization of the patch grid.
    """
    if masked_data_array is None:
        print("      -> Masked data is None. Skipping patch creation.")
        return

    print(f"    Slicing {date_str} data into {patch_size}x{patch_size} patches...")
    height, width, num_channels = masked_data_array.shape

    # Calculate the number of full patches that can be extracted
    patches_y = height // patch_size
    patches_x = width // patch_size
    num_patches = patches_y * patches_x

    if num_patches == 0:
        print("      -> No full patches could be extracted. Skipping file save.")
        return

    # --- MEMORY-EFFICIENT PRE-ALLOCATION ---
    print(f"      -> Pre-allocating memory for {num_patches} patches...")
    try:
        patches_array = np.zeros((num_patches, patch_size, patch_size, num_channels), dtype=np.float32)
    except MemoryError:
        print("      ERROR: Not enough RAM to pre-allocate memory for all patches.")
        return
    
    patch_index = 0
    # Loop and copy slices directly into the pre-allocated array
    for y in range(patches_y):
        for x in range(patches_x):
            start_y = y * patch_size
            start_x = x * patch_size
            patch = masked_data_array[start_y : start_y + patch_size, start_x : start_x + patch_size, :]
            patches_array[patch_index] = patch
            patch_index += 1

    # --- Print Final Data Stats ---
    print_raster_stats(patches_array, f"{date_str} Final Patches")
    
    # --- CRITICAL FIX: Save as a .mat file ---
    # The data will be saved inside the .mat file under the variable name 'patches'
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

