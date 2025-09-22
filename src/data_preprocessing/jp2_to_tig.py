"""
Task 0: Convert Sentinel-2 JP2 files to GeoTIFF format.

This script runs first to solve the rasterio driver issue on Kaggle
by converting all problematic .jp2 files into a universally readable
.tif format using the system's gdal_translate command.
"""
import os
import subprocess
import glob

def convert_all_jp2_to_tif(raw_data_dir):
    """
    Scans the raw data directory for .jp2 files and converts them.
    """
    print("--- Stage 0: Converting all Sentinel-2 JP2 files to GeoTIFF ---")
    print("This may take several minutes...")
    
    # Use glob to recursively find all .jp2 files
    jp2_files = glob.glob(os.path.join(raw_data_dir, '**', '*.jp2'), recursive=True)
    
    if not jp2_files:
        print("  -> No .jp2 files found to convert.")
        return

    converted_count = 0
    for jp2_path in jp2_files:
        # Create the output filename by replacing the extension
        tif_path = os.path.splitext(jp2_path)[0] + '.tif'
        
        # Skip if the converted file already exists
        if os.path.exists(tif_path):
            continue
            
        # Use the powerful gdal_translate command-line tool
        command = [
            'gdal_translate',
            '-of', 'GTiff',   # Output format: GeoTIFF
            jp2_path,         # Input file
            tif_path          # Output file
        ]
        
        try:
            # Run the command
            subprocess.run(command, check=True, capture_output=True, text=True)
            converted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ERROR converting {os.path.basename(jp2_path)}.")
            print(f"  GDAL Error: {e.stderr}")
        except FileNotFoundError:
            print("FATAL ERROR: gdal_translate not found. This shouldn't happen on Kaggle.")
            return

    print(f"  -> Conversion complete. Converted {converted_count} files.")
