"""
Main script to orchestrate the model training process.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import joblib

# Import from our source library
from src.config import globals as config
from src.dataset.dataset import LocalSequenceDataset
from src.models.cnn_encoder import ResNetEncoder
from src.models.seq2seq_model import MultiModalSeq2Seq
from src.training.trainer import train_model

def setup_preprocessing():
    """Generates IoT data and fits scalers/encoders."""
    print("--- Setting up preprocessing objects ---")
    # ... (Code to generate IoT data and fit scalers/encoders) ...
    return iot_data, scalers, encoders

if __name__ == '__main__':
    # --- 1. Setup ---
    os.makedirs(config.OUTPUT_MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 2. Preprocessing Objects ---
    # In a real project, you would have a more robust way to handle this.
    # For now, we'll keep the generation logic here.
    iot_data, all_iot_dfs = {}, []
    for event in config.EVENT_METADATA:
        # Generate dummy data for now
        iot_data[event] = np.random.rand(config.N_STEPS_IN + config.N_STEPS_OUT, 3) 
    
    # Dummy scalers and encoders
    all_crops = [[meta['crop_type']] for meta in config.EVENT_METADATA.values()]
    all_diseases = [[meta['disease']] for meta in config.EVENT_METADATA.values()]
    scalers = {'iot': StandardScaler().fit(np.random.rand(100, 3))}
    encoders = {
        'crop': OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(all_crops),
        'disease': OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(all_diseases)
    }
    
    # --- 3. Data Loading ---
    full_dataset = LocalSequenceDataset(config.INPUT_DATA_DIR, config.EVENT_METADATA, iot_data, scalers, encoders)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Created train ({len(train_dataset)}) and validation ({len(val_dataset)}) sets.")

    # --- 4. Model & Optimizer ---
    num_tabular_features = 3 + len(encoders['crop'].categories_[0]) + len(encoders['disease'].categories_[0])
    num_classes = len(set(meta['label'] for meta in config.EVENT_METADATA.values()))
    
    cnn = ResNetEncoder(feature_vector_size=config.FEATURE_VECTOR_SIZE).to(device)
    model = MultiModalSeq2Seq(cnn, num_tabular_features, num_classes).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- 5. Loss Functions with Class Weights ---
    all_labels = [config.EVENT_METADATA[s['event']]['label'] for s in train_dataset.dataset.samples[train_dataset.indices]]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    health_criterion = nn.MSELoss()
    
    # --- 6. Start Training ---
    print("\n--- Starting Full Training and Validation Loop ---")
    train_model(model, train_loader, val_loader, optimizer, class_criterion, health_criterion, device)
    
    print("\n--- TRAINING COMPLETE ---")
