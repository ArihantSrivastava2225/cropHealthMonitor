"""
Contains the main training and validation loop logic.
"""
import torch
import os
from torch.amp import GradScaler, autocast
from src.config import globals as config

def train_model(model, train_loader, val_loader, optimizer, class_criterion, health_criterion, device):
    scaler = GradScaler()
    best_val_accuracy = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for (X_img_b, X_tab_b), (y_class_b, y_health_b) in train_loader:
            X_tab_b = X_tab_b.to(device)
            y_class_b = y_class_b.to(device)
            y_health_b = y_health_b.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                y_class_pred, y_health_pred = model(X_img_b, X_tab_b)
                loss_class = class_criterion(y_class_pred, y_class_b)
                loss_health = health_criterion(y_health_pred, y_health_b)
                loss = loss_class + (config.HEALTH_LOSS_WEIGHT * loss_health)
            
            if not torch.isfinite(loss):
                print(f"WARNING: Skipping update at epoch {epoch} due to non-finite loss.")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # --- Validation Loop ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for (X_img_b, X_tab_b), (y_class_b, y_health_b) in val_loader:
                X_tab_b, y_class_b, y_health_b = X_tab_b.to(device), y_class_b.to(device), y_health_b.to(device)
                with autocast(device_type="cuda"):
                    y_class_pred, y_health_pred = model(X_img_b, X_tab_b)
                    loss_class = class_criterion(y_class_pred, y_class_b)
                    loss_health = health_criterion(y_health_pred, y_health_b)
                    loss = loss_class + (config.HEALTH_LOSS_WEIGHT * loss_health)
                
                val_loss += loss.item()
                _, predicted = torch.max(y_class_pred.data, 1)
                total += y_class_b.size(0)
                correct += (predicted == y_class_b).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch:02d}/{config.EPOCHS}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_MODEL_DIR, 'best_crop_model.pth'))
            print(f"  -> New best model saved with accuracy: {best_val_accuracy:.2f}%")
