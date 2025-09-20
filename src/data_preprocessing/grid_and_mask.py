"""
Task 1: Scans all satellite products for an event, determines the total
geographic extent to define a universal common grid, and fetches a single
cropland mask for that entire grid.
"""
import os
import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import numpy as np
from PIL import Image
import pystac_client
import planetary_computer
from rasterio.enums import Resampling
from rasterio.warp import reproject

def define_event_grid_and_mask(event_raw_dir, viz_dir, event_name, target_resolution):
    """Finds the union of all product bounds and creates a single grid and mask."""
    print("Step 1: Defining a universal grid for the entire event...")
    
    # --- Find all reference bands to determine total extent ---
    all_ref_bands = []
    date_folders = [d for d in os.listdir(event_raw_dir) if os.path.isdir(os.path.join(event_raw_dir, d))]
    for date_folder in date_folders:
        date_folder_path = os.path.join(event_raw_dir, date_folder)
        product_folders = [os.path.join(date_folder_path, d) for d in os.listdir(date_folder_path) if os.path.isdir(os.path.join(date_folder_path, d))]
        for product_path in product_folders:
            ref_band_path = None
            try:
                if "S2" in os.path.basename(product_path):
                    granule = [f for f in os.listdir(os.path.join(product_path, 'GRANULE')) if f.startswith('L2A')][0]
                    parts = os.path.basename(product_path).split('_'); tile_id = parts[5]; date_time = parts[2]
                    ref_band_path = os.path.join(product_path, 'GRANULE', granule, 'IMG_DATA', 'R10m', f'{tile_id}_{date_time}_B04_10m.jp2')
                else: # Landsat
                    all_files = os.listdir(product_path)
                    ref_band_filename = [f for f in all_files if f.endswith('_SR_B4.TIF')][0]
                    ref_band_path = os.path.join(product_path, ref_band_filename)
                
                if ref_band_path and os.path.exists(ref_band_path):
                    all_ref_bands.append(ref_band_path)
            except (IndexError, FileNotFoundError):
                print(f"  - Warning: Could not find a reference band in {os.path.basename(product_path)}")
                continue
    
    if not all_ref_bands:
        print("  ERROR: No valid satellite products with reference bands found. Cannot define grid.")
        return None, None

    # --- Calculate the union of all bounds ---
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    target_crs = None
    for band_path in all_ref_bands:
        with rasterio.open(band_path) as src:
            min_x = min(min_x, src.bounds.left)
            min_y = min(min_y, src.bounds.bottom)
            max_x = max(max_x, src.bounds.right)
            max_y = max(max_y, src.bounds.top)
            if target_crs is None:
                target_crs = src.crs

    # --- Create the common grid based on the union ---
    target_transform = rasterio.transform.from_origin(min_x, max_y, target_resolution, target_resolution)
    target_height = int(np.ceil((max_y - min_y) / target_resolution))
    target_width = int(np.ceil((max_x - min_x) / target_resolution))
    target_shape = (target_height, target_width)
    common_grid = (target_crs, target_transform, target_shape)
    print(f"  -> Universal grid created with shape: {target_shape}")

    # --- Fetch Cropland Mask for the entire grid ---
    print("Step 2: Fetching universal cropland mask...")
    wgs84_bounds = transform_bounds(target_crs, CRS.from_epsg(4326), min_x, min_y, max_x, max_y)
    
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=["io-lulc-9-class"], bbox=wgs84_bounds)
    
    # CRITICAL FIX: Use the .items() iterator to handle large queries page by page
    items_iterator = search.items()
    first_item = next(items_iterator, None) # Get the first result

    if not first_item:
        print("  -> WARNING: No Land Use/Land Cover data found. Using a full mask.")
        cropland_mask = np.ones(target_shape, dtype=bool)
    else:
        lulc_href = first_item.assets["data"].href
        with rasterio.open(lulc_href) as lulc_src:
            mask_reprojected = np.empty(target_shape, dtype=lulc_src.dtypes[0])
            reproject(
                source=rasterio.band(lulc_src, 1), destination=mask_reprojected,
                src_transform=lulc_src.transform, src_crs=lulc_src.crs,
                dst_transform=target_transform, dst_crs=target_crs,
                resampling=Resampling.nearest
            )
        cropland_mask = (mask_reprojected == 5)
        print(f"  -> SUCCESS: Cropland mask fetched. {np.sum(cropland_mask) / cropland_mask.size:.2%} of the area is cropland.")

    # --- Visualization ---
    mask_viz_path = os.path.join(viz_dir, f"{event_name}_00_universal_cropland_mask.png")
    mask_img_array = cropland_mask.astype(np.uint8) * 255
    img = Image.fromarray(mask_img_array, 'L')
    img.save(mask_viz_path)
    print(f"  -> Mask visualization saved to: {os.path.basename(mask_viz_path)}")
    
    return common_grid, cropland_mask

