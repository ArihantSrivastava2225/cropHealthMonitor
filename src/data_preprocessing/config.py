"""
Central Configuration File for the Preprocessing Pipeline
*** THIS VERSION IS CONFIGURED FOR THE KAGGLE NOTEBOOK ENVIRONMENT ***
"""
import os

# --- Base Paths for Kaggle ---
# Input data comes from the read-only /kaggle/input/ directory.
# Replace 'crophealthsatellitedatav1' with the exact name of your Kaggle dataset.
RAW_DATA_DIR = '/kaggle/input/crophealthsatellitedatav1/organized_date'

# All output must be written to the /kaggle/working/ directory.
PROCESSED_DATA_DIR = '/kaggle/working/processed'

# --- Processing Settings (remain the same) ---
PATCH_SIZE = 256
TARGET_RESOLUTION = 30  # meters

# --- Metadata (must match folder names in your input data) ---
EVENT_METADATA = {
    'Bathinda-PinkBollworm': {},
    'EasternUP-RedRot': {},
    'Haryana-RiceBlast': {}, # Added from your screenshot
    'Punjab-leafhopper': {},
    'Ropar-wheatRust': {},
    'Una-yellowRust': {}
    # Add any other event folders you have
}

### 2. How to Run the Pipeline on Kaggle

# Now that your code is ready, here are the steps and the exact commands to run in your Kaggle notebook.

# **Step 1: Create a New Kaggle Notebook**
# * Go to Kaggle and start a new notebook.

# **Step 2: Add Your Data and Code**
# * **Add Your Raw Data:** In the notebook editor, click on **"+ Add Input"**. Go to the "Your Datasets" tab and add your `crophealthsatellitedatav1` dataset.
# * **Clone Your GitHub Repo:** Click **"+ Add Input"** again. Go to the "Git" tab, enter the URL of your GitHub repository, and click the arrow to clone it. It will appear in `/kaggle/working/` with your repository's name (e.g., `cropHealthMonitor`).

# **Step 3: Install Dependencies**
# * In the first code cell of your notebook, run the following command to install all the necessary libraries from your `requirements.txt` file.

# ```bash
# !pip install -q -r /kaggle/working/cropHealthMonitor/requirements.txt
# ```

# **Step 4: Run the Preprocessing Pipeline**
# * In the second code cell, run the command to execute your main pipeline script. This command tells Python to run your orchestrator, which will then call all the other modular scripts in the correct order.

# ```bash
# !python /kaggle/working/cropHealthMonitor/src/data_preprocessing/run_pipeline.py
