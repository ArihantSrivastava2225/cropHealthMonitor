"""
Main script to run inference and generate health maps for a specific event.

This script acts as the main entry point for the inference pipeline. It
loads the trained model and preprocessing objects, prepares the data for a
specified event, runs predictions, and generates the final visual maps.

Example usage from the terminal in the project's root directory:
> python inference.py --event Bathinda-PinkBollworm
"""
import os
import torch
import joblib
import argparse
import numpy as np
from torch.utils.data import DataLoader

# Import from our source code library
from src.config import globals as config
from src.dataset.dataset import InferenceDataset
from src.models.cnn_encoder import ResNetEncoder
from src.models.seq2seq_model import MultiModalSeq2Seq
from src.inference.predictor import run_predictions
from src.inference.map_generator import generate_maps

def main(event_name):
    """Orchestrates the entire inference process for a given event."""
    print(f"--- Starting inference for event: {event_name} ---")
    
    # --- 1. Setup Environment ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = os.path.join(config.OUTPUT_MODEL_DIR, 'best_crop_model.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please run train.py first.")
        return

    # --- 2. Load Preprocessing Objects ---
    print("Loading preprocessing objects...")
    # NOTE: In a real-world scenario, you would generate or load the actual IoT data
    # that corresponds to the timeframe of your inference data. For this project,
    # we are re-using the dummy data generated during training.
    iot_data = {}
    for event in config.EVENT_METADATA:
        # This part assumes dummy/simulated data is available for all events.
        # In a real application, you would load the specific data for the `event_name`.
        iot_data[event] = np.random.rand(config.N_STEPS_IN + config.N_STEPS_OUT, 3) 

    try:
        scalers = {'iot': joblib.load(os.path.join(config.OUTPUT_MODEL_DIR, 'iot_scaler.gz'))}
        encoders = {
            'crop': joblib.load(os.path.join(config.OUTPUT_MODEL_DIR, 'crop_encoder.gz')),
            'disease': joblib.load(os.path.join(config.OUTPUT_MODEL_DIR, 'disease_encoder.gz'))
        }
    except FileNotFoundError:
        print("ERROR: Preprocessing files (scaler/encoders) not found. Please run train.py to generate them.")
        return
    
    # --- 3. Prepare Data Loader ---
    inference_dataset = InferenceDataset(config.INPUT_DATA_DIR, event_name, iot_data, scalers, encoders)
    if len(inference_dataset) == 0:
        print(f"No data found for event '{event_name}' in '{config.INPUT_DATA_DIR}'. Exiting.")
        return
    
    inference_loader = DataLoader(inference_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # --- 4. Load Trained Model ---
    print("Loading trained model architecture and weights...")
    num_tabular_features = 3 + len(encoders['crop'].categories_[0]) + len(encoders['disease'].categories_[0])
    num_classes = len(set(meta['label'] for meta in config.EVENT_METADATA.values()))
    
    cnn = ResNetEncoder(feature_vector_size=config.FEATURE_VECTOR_SIZE)
    model = MultiModalSeq2Seq(cnn, num_tabular_features, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # --- 5. Run Predictions ---
    class_preds, health_preds = run_predictions(model, inference_loader, device)
    
    # --- 6. Generate and Save Maps ---
    generate_maps(class_preds, health_preds, encoders['disease'], event_name, config.OUTPUT_MODEL_DIR)

if __name__ == '__main__':
    # --- Argument Parser to select the event from the command line ---
    parser = argparse.ArgumentParser(description="Generate health maps for a specific crop health event.")
    parser.add_argument('--event', type=str, required=True, 
                        choices=config.EVENT_METADATA.keys(),
                        help='The name of the event folder to process.')
    args = parser.parse_args()
    
    main(args.event)

# ### How to Run This Script

# After you have successfully run `train.py` and have a `best_crop_model.pth` file in your `saved_models` directory:

# 1.  Open your terminal or command prompt.
# 2.  Make sure you are in the root of your project folder (e.g., `crop_health_project/`).
# 3.  Run the script by typing `python inference.py` followed by `--event` and the name of the event folder you want to analyze.

# **Example Command:**
# ```bash
# python inference.py --event Bathinda-PinkBollworm