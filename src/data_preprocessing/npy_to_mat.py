"""
NPY to MAT File Converter

This script converts all .npy files within a specified input directory
to .mat files in an output directory, preserving the folder structure.

It saves the data from each .npy file under a specific variable
name inside the corresponding .mat file, making it easy to load in MATLAB.
"""

import os
import numpy as np
import scipy.io
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# --- Set your input and output directories here ---

# The folder containing your event subfolders with .npy files
# Example: 'crop_health_project/data/processed_npy'
INPUT_DIR = 'path/to/your/npy_folder' 

# The folder where the new .mat files will be saved
# Example: 'crop_health_project/data/processed_mat'
OUTPUT_DIR = 'path/to/your/mat_folder'

# The variable name to use inside the .mat file.
# When you load the file in MATLAB, the data will be in this variable.
# For our project, 'patches' is the expected name.
MATLAB_VARIABLE_NAME = 'patches'

# ==============================================================================
# MAIN CONVERSION LOGIC
# ==============================================================================

def convert_npy_to_mat(input_dir, output_dir, var_name):
    """
    Walks through the input directory, converts .npy to .mat, and saves
    in the output directory, replicating the folder structure.
    """
    print(f"Starting conversion from '{input_dir}' to '{output_dir}'...")
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        sys.exit(1)

    # Use os.walk to go through all subdirectories
    for dirpath, _, filenames in os.walk(input_dir):
        # Create the corresponding output directory structure
        relative_path = os.path.relpath(dirpath, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(output_path, exist_ok=True)
        
        for filename in filenames:
            if filename.endswith('.npy'):
                npy_file_path = os.path.join(dirpath, filename)
                
                # Create the new .mat filename
                base_filename = os.path.splitext(filename)[0]
                mat_file_path = os.path.join(output_path, f"{base_filename}.mat")
                
                try:
                    # 1. Load the .npy file
                    data = np.load(npy_file_path)
                    
                    # 2. Create the dictionary for the .mat file
                    # The key is the variable name that will appear in MATLAB
                    mat_dict = {var_name: data}
                    
                    # 3. Save as a .mat file
                    # do_compression=True is good practice for large files
                    scipy.io.savemat(mat_file_path, mat_dict, do_compression=True)
                    
                    print(f"  Converted '{npy_file_path}' -> '{mat_file_path}'")

                except Exception as e:
                    print(f"  ERROR: Could not convert '{npy_file_path}'. Reason: {e}")

    print("\nConversion complete!")


if __name__ == '__main__':
    # Make sure to update the INPUT_DIR and OUTPUT_DIR paths above
    # before running the script.
    if INPUT_DIR == 'path/to/your/npy_folder' or OUTPUT_DIR == 'path/to/your/mat_folder':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE THE INPUT_DIR AND OUTPUT_DIR PATHS !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        convert_npy_to_mat(INPUT_DIR, OUTPUT_DIR, MATLAB_VARIABLE_NAME)

