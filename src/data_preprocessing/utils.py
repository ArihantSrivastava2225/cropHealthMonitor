"""
Utility functions shared across the preprocessing pipeline.
This version uses memory-efficient, NaN-ignoring functions for statistics.
"""
import numpy as np
import os

def print_raster_stats(data_array, name=""):
    """
    Prints summary statistics for a raster data array in a memory-efficient,
    band-by-band or patch-by-patch manner.
    """
    if data_array is None:
        print(f"  STATS for {name}: Data is None")
        return
        
    print(f"  STATS for {name}:")
    print(f"    - Shape: {data_array.shape}")
    
    # Check if the array is multi-band
    if data_array.ndim == 3: # (H, W, C)
        num_bands = data_array.shape[2]
        # Calculate stats for each band individually to save RAM
        for i in range(num_bands):
            channel = data_array[:, :, i]
            # Use NaN-ignoring functions to avoid large temp boolean arrays
            mean_val = np.nanmean(channel)
            max_val = np.nanmax(channel)
            min_val = np.nanmin(channel)
            std_val = np.nanstd(channel)
            print(f"    - Band {i+1}: Mean={mean_val:.4f}, Max={max_val:.4f}, Min={min_val:.4f}, Std={std_val:.4f}")
    elif data_array.ndim == 4: # (N, H, W, C) for patches
        # For large patch arrays, just stats on a sample patch is enough
        print("    - (Stats for patches are calculated on a sample)")
        sample_patch = data_array[0, :, :, :]
        # Here nan_to_num is safe as a single patch is very small
        stats_array = np.nan_to_num(sample_patch)
        print(f"    - Sample Patch: Mean={np.mean(stats_array):.4f}, Max={np.max(stats_array):.4f}, Min={np.min(stats_array):.4f}, Std={np.std(stats_array):.4f}")
    else: # For 2D or other arrays
        mean_val = np.nanmean(data_array)
        max_val = np.nanmax(data_array)
        min_val = np.nanmin(data_array)
        std_val = np.nanstd(data_array)
        print(f"    - Overall: Mean={mean_val:.4f}, Max={max_val:.4f}, Min={min_val:.4f}, Std={std_val:.4f}")


def create_false_color_composite(red, nir, swir1):
    """Creates a visually intuitive false-color image (vegetation is red)."""
    def normalize(band):
        """A memory-efficient normalization function for visualization."""
        band = np.nan_to_num(band) # nan_to_num is fine here as it's called on a single band
        non_zero_vals = band[band > 0]
        if non_zero_vals.size == 0:
            return np.zeros_like(band, dtype=np.uint8)
        
        p2, p98 = np.percentile(non_zero_vals, (2, 98))
        p2, p98 = np.float32(p2), np.float32(p98)
        band_32 = band.astype(np.float32)
        clipped = np.clip(band_32, p2, p98)
        
        if p98 > p2:
            normalized = (clipped - p2) / (p98 - p2)
            return (normalized * 255).astype(np.uint8)
        return np.zeros_like(band, dtype=np.uint8)

    return np.stack([normalize(swir1), normalize(nir), normalize(red)], axis=-1)

