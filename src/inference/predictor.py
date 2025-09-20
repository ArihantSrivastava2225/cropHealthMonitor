"""
Contains the core logic for running the trained model on a dataset
to generate predictions.
"""
import torch
import numpy as np
from tqdm import tqdm

def run_predictions(model, data_loader, device):
    """
    Runs the model over all data in the data_loader and collects predictions.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the inference dataset.
        device (torch.device): The device to run inference on (e.g., 'cuda').

    Returns:
        tuple: A tuple containing two lists:
               - all_class_preds (list): Predicted class labels.
               - all_health_preds (list): Predicted health index values.
    """
    model.eval()
    all_class_preds = []
    all_health_preds = []

    with torch.no_grad():
        for (X_img_b, X_tab_b) in tqdm(data_loader, desc="Generating Predictions"):
            X_tab_b = X_tab_b.to(device)
            
            # The model expects image data on the CPU
            y_class_pred, y_health_pred = model(X_img_b, X_tab_b)
            
            _, predicted_class = torch.max(y_class_pred.data, 1)
            
            all_class_preds.extend(predicted_class.cpu().numpy())
            all_health_preds.extend(y_health_pred.cpu().numpy())
            
    return all_class_preds, all_health_preds
