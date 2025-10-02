"""
Task 3: Reads individual band files from disk and incrementally builds
the final 6-channel patch array, saving each valid patch as a separate .mat file.
This approach is highly memory-efficient for very large datasets.
"""
import os
import numpy as np
import scipy.io
from PIL import Image, ImageDraw
from .utils import print_raster_stats

# --- CRITICAL FIX: Disable the Decompression Bomb check for large images ---
# Setting this to None allows Pillow to open images of any size.
Image.MAX_IMAGE_PIXELS = None

def create_and_save_individual_patches(temp_band_paths, date_str, patch_size, output_dir_for_patches, output_viz_dir):
    """
    Builds and saves individual patch files by reading from temporary band files.
    """
    if not temp_band_paths:
        print("      -> No temporary band files found. Skipping patch creation.")
        return

    print(f"    Slicing {date_str} data into individual {patch_size}x{patch_size} patch files...")
    
    try:
        # Load the first band to get the dimensions of the full image
        first_band = np.load(temp_band_paths[0])
        height, width = first_band.shape
        num_channels = len(temp_band_paths)
    except Exception as e:
        print(f"      ERROR: Could not load temporary band file to get dimensions. {e}")
        return
        
    patches_y = height // patch_size
    patches_x = width // patch_size
    
    if patches_y * patches_x == 0:
        print("      -> No full patches could be extracted. Skipping.")
        return

    saved_patch_count = 0
    # Load NIR band (index 3) once to efficiently check for valid patches
    nir_band_data = np.load(temp_band_paths[3])
    
    # Loop through the grid of potential patch locations
    for y_idx in range(patches_y):
        for x_idx in range(patches_x):
            start_y, start_x = y_idx * patch_size, x_idx * patch_size
            
            # Use the pre-loaded NIR band to check if this patch is valid cropland
            nir_patch_slice = nir_band_data[start_y:start_y+patch_size, start_x:start_x+patch_size]
            
            # If the patch is more than 90% empty (non-cropland), we skip it
            if np.count_nonzero(nir_patch_slice) / nir_patch_slice.size < 0.1:
                continue

            # --- If the patch is valid, we build the full 6-channel data for it ---
            
            # Pre-allocate memory for just one small 6-channel patch
            full_patch = np.zeros((patch_size, patch_size, num_channels), dtype=np.float32)
            
            for c, band_path in enumerate(temp_band_paths):
                # Load the full band from disk
                band_data = np.load(band_path)
                # Slice out just the small piece we need for this patch
                full_patch[:, :, c] = band_data[start_y:start_y+patch_size, start_x:start_x+patch_size]
            
            # Save this single, complete patch to its own .mat file
            patch_filename = f"patch_{y_idx}_{x_idx}.mat"
            output_mat_path = os.path.join(output_dir_for_patches, patch_filename)
            scipy.io.savemat(output_mat_path, {'patch_data': full_patch})
            saved_patch_count += 1
            
    print(f"      -> Filtered and saved {saved_patch_count} valid cropland patches to '{date_str}' directory.")

    # --- Create Grid Visualization (shows ALL potential patch locations) ---
    # We only create the visualization if at least one patch was saved to avoid errors
    if saved_patch_count > 0:
        base_image_path = os.path.join(output_viz_dir, f"{date_str}_02_after_mask.png")
        if os.path.exists(base_image_path):
            try:
                img = Image.open(base_image_path).convert("RGBA")
                draw = ImageDraw.Draw(img)
                for y in range(patches_y):
                    for x in range(patches_x):
                        start_y = y * patch_size
                        start_x = x * patch_size
                        # Draw a rectangle for every potential patch location on the visualization
                        draw.rectangle([start_x, start_y, start_x + patch_size - 1, start_y + patch_size - 1], outline="cyan", width=2)
                
                viz_path = os.path.join(output_viz_dir, f"{date_str}_03_patch_grid.png")
                img.save(viz_path)
                print(f"      -> Patch grid visualization saved.")
            except Exception as e:
                print(f"      WARNING: Could not create grid visualization. Reason: {e}")

