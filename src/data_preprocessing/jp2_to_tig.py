"""
Task 0: Prepares a writable copy of the raw dataset.

This script is the definitive fix for Kaggle's read-only file system and
rasterio driver issues. It walks the read-only /kaggle/input directory,
recreates the entire folder structure in the writable /kaggle/working
directory, and converts any problematic .jp2 files to .tif format
during the copy process.
"""
import os
import shutil
import subprocess
import glob

def prepare_writable_dataset(input_dir, output_dir):
    """
    Copies the dataset from a read-only to a writable location,
    converting .jp2 files to .tif along the way.
    """
    print("--- Stage 0: Preparing a Writable Dataset Copy ---")
    print(f"Reading from: {input_dir}")
    print(f"Writing to: {output_dir}")
    
    if os.path.exists(output_dir):
        print("  -> Writable data directory already exists. Skipping copy/conversion.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    converted_count = 0
    copied_count = 0

    # Walk through the entire read-only input directory structure
    for dirpath, dirnames, filenames in os.walk(input_dir):
        # Create the corresponding directory structure in the output
        relative_path = os.path.relpath(dirpath, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(output_path, exist_ok=True)
        
        for filename in filenames:
            source_file = os.path.join(dirpath, filename)
            dest_file = os.path.join(output_path, filename)

            if filename.endswith('.jp2'):
                # For .jp2 files, convert them to .tif in the destination
                tif_path = os.path.splitext(dest_file)[0] + '.tif'
                if os.path.exists(tif_path): continue

                command = ['gdal_translate', '-of', 'GTiff', source_file, tif_path]
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    converted_count += 1
                except Exception as e:
                    print(f"  ERROR converting {filename}: {e}")
            else:
                # For all other files, just copy them
                if os.path.exists(dest_file): continue
                shutil.copy2(source_file, dest_file)
                copied_count += 1
                
    print(f"  -> Conversion complete. Converted {converted_count} JP2 files.")
    print(f"  -> Copied {copied_count} other files.")
