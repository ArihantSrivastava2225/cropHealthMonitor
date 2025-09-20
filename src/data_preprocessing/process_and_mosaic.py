"""
Task 2: Processes all satellite products for a single day using a memory-efficient
iterative mosaicking approach and in-place normalization to handle very large
geographic areas.
"""
import os
import glob
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject
from PIL import Image
from utils import create_false_color_composite, print_raster_stats

def process_and_mosaic_daily_data(product_paths, common_grid, cropland_mask, viz_dir, date_str):
    """Processes all products for one day and returns a final 6-channel masked array."""
    target_crs, target_transform, target_shape = common_grid
    
    # --- MEMORY-EFFICIENT MOSAICKING SETUP ---
    mosaic_canvas_5_band = np.zeros(target_shape + (5,), dtype=np.float32)
    count_canvas = np.zeros(target_shape, dtype=np.uint8)

    print("    Iteratively processing and mosaicking daily products...")
    for product_path in product_paths:
        print(f"    - Processing product: {os.path.basename(product_path)}")
        
        # --- Define Band Mapping ---
        if "S2" in os.path.basename(product_path):
            band_map = {'blue': 'B02', 'green': 'B03', 'red': 'B04', 'nir': 'B08', 'swir1': 'B11'}
        else:
            band_map = {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4', 'nir': 'SR_B5', 'swir1': 'SR_B6'}

        # --- Process one product at a time ---
        reprojected_bands = {}
        if "S2" in os.path.basename(product_path):
            granule_path_list = glob.glob(os.path.join(product_path, 'GRANULE', 'L2A*'))
            if not granule_path_list: continue
            granule_path = granule_path_list[0]
            for band_name, s2_band_code in band_map.items():
                res_folder = 'R20m' if s2_band_code in ['B11'] else 'R10m'
                search_pattern = os.path.join(granule_path, 'IMG_DATA', res_folder, f'*_{s2_band_code}_*.jp2')
                try:
                    file_path = glob.glob(search_pattern)[0]
                    with rasterio.open(file_path) as src:
                        destination = np.empty(target_shape, dtype=np.float32)
                        reproject(source=rasterio.band(src, 1), destination=destination, src_transform=src.transform, src_crs=src.crs, dst_transform=target_transform, dst_crs=target_crs, resampling=Resampling.bilinear)
                        reprojected_bands[band_name] = destination
                except IndexError: continue
        else: # Landsat
            all_files = os.listdir(product_path)
            for band_name, l8_band_code in band_map.items():
                try:
                    band_filename = [f for f in all_files if f.endswith(f'_{l8_band_code}.TIF')][0]
                    file_path = os.path.join(product_path, band_filename)
                    with rasterio.open(file_path) as src:
                        destination = np.empty(target_shape, dtype=np.float32)
                        reproject(source=rasterio.band(src, 1), destination=destination, src_transform=src.transform, src_crs=src.crs, dst_transform=target_transform, dst_crs=target_crs, resampling=Resampling.bilinear)
                        reprojected_bands[band_name] = destination
                except IndexError: continue
        
        if len(reprojected_bands) == 5:
            current_product_5_band = np.stack(list(reprojected_bands.values()), axis=-1)
            valid_data_mask = np.any(current_product_5_band != 0, axis=2)
            mosaic_canvas_5_band[valid_data_mask] += current_product_5_band[valid_data_mask]
            count_canvas[valid_data_mask] += 1
    
    # --- Finalize the Mosaic ---
    count_canvas_expanded = np.expand_dims(count_canvas, axis=2)
    count_canvas_expanded[count_canvas_expanded == 0] = 1
    final_mosaic_5_band = mosaic_canvas_5_band / count_canvas_expanded
    print_raster_stats(final_mosaic_5_band, f"{date_str} Raw Mosaic (5 bands)")
    del mosaic_canvas_5_band, count_canvas, count_canvas_expanded

    # --- Create "Before" Visualization ---
    red_before, nir_before, swir1_before = [final_mosaic_5_band[:,:,i] for i in [2, 3, 4]]
    false_color_before = create_false_color_composite(red_before, nir_before, swir1_before)
    Image.fromarray(false_color_before).save(os.path.join(viz_dir, f"{date_str}_01_before_mask.png"))
    print("      -> 'Before' visualization saved.")
    
    # --- Apply Mask to the 5 raw bands ---
    masked_5_band_array = final_mosaic_5_band * cropland_mask[..., np.newaxis]
    del final_mosaic_5_band
    
    # --- NORMALIZE THE MASKED BANDS IN-PLACE ---
    print("    Normalizing masked data (memory-efficiently)...")
    for i in range(5):
        channel = masked_5_band_array[:, :, i]
        non_zero_vals = channel[channel > 0]
        if non_zero_vals.size > 0:
            p2, p98 = np.percentile(non_zero_vals, (2, 98))
            np.clip(channel, p2, p98, out=channel)
            if p98 > p2:
                mask = channel > 0
                channel[mask] = (channel[mask] - p2) / (p98 - p2)
    
    # --- Calculate Indices from CLEAN, NORMALIZED, MASKED data ---
    blue, green, red, nir, swir1 = [masked_5_band_array[:,:,i] for i in range(5)]
    np.seterr(divide='ignore', invalid='ignore')
    
    # --- CRITICAL FIX: Memory-Efficient Index Calculation ---
    print("    Calculating indices (memory-efficiently)...")
    
    # Calculate NDVI one step at a time to conserve RAM
    numerator_ndvi = nir - red
    denominator_ndvi = nir + red
    denominator_ndvi[denominator_ndvi == 0] = 1 # Avoid division by zero
    ndvi = numerator_ndvi / denominator_ndvi
    del numerator_ndvi, denominator_ndvi # Free memory immediately

    # Calculate NDMI one step at a time
    numerator_ndmi = nir - swir1
    denominator_ndmi = nir + swir1
    denominator_ndmi[denominator_ndmi == 0] = 1
    ndmi = numerator_ndmi / denominator_ndmi
    del numerator_ndmi, denominator_ndmi # Free memory immediately

    ndvi = np.nan_to_num(np.clip(ndvi, -1, 1))
    ndmi = np.nan_to_num(np.clip(ndmi, -1, 1))

    # --- Create "After" Visualization ---
    false_color_after = create_false_color_composite(red, nir, swir1)
    Image.fromarray(false_color_after).save(os.path.join(viz_dir, f"{date_str}_02_after_mask.png"))
    print("      -> 'After' visualization saved.")
    
    # --- Stack all clean, normalized bands at the very end ---
    final_masked_data = np.stack([blue, green, red, nir, ndvi, ndmi], axis=-1)
    print_raster_stats(final_masked_data, f"{date_str} Final Normalized Data (6 bands)")
    
    return final_masked_data

