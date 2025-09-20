"""
Contains the logic for generating and saving the final health map visualizations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_maps(class_preds, health_preds, disease_encoder, event_name, output_dir):
    """
    Generates and saves the Disease Risk and Crop Stress maps.
    """
    print("\n--- Reconstructing and Generating Health Maps ---")
    
    # --- Reconstruct the Map Grid ---
    total_patches = len(class_preds)
    if total_patches == 0:
        print("No predictions to map. Exiting.")
        return
        
    patches_per_row = int(np.sqrt(total_patches))
    if patches_per_row * patches_per_row != total_patches:
        print(f"Warning: The number of patches ({total_patches}) does not form a perfect square.")
        patches_per_row = int(np.floor(np.sqrt(total_patches)))
        while patches_per_row > 0 and total_patches % patches_per_row != 0:
            patches_per_row -= 1
    
    if patches_per_row == 0:
        print("Could not determine map dimensions.")
        return

    num_rows = total_patches // patches_per_row
    
    class_map = np.array(class_preds).reshape(num_rows, patches_per_row)
    health_map = np.array(health_preds).reshape(num_rows, patches_per_row)

    # --- Visualize the Maps ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(f"Analysis for Event: {event_name}", fontsize=16)
    
    # 1. Disease Risk Map
    unique_labels = sorted(list(set(class_preds)))
    cmap_risk = plt.get_cmap('viridis', len(unique_labels))
    im1 = axes[0].imshow(class_map, cmap=cmap_risk)
    axes[0].set_title('Predicted Disease Risk Map')
    axes[0].set_xlabel('Patch Column')
    axes[0].set_ylabel('Patch Row')
    
    cbar1 = fig.colorbar(im1, ax=axes[0], ticks=unique_labels)
    disease_names = disease_encoder.categories_[0]
    tick_labels = [disease_names[label] for label in unique_labels]
    cbar1.set_ticklabels(tick_labels)

    # 2. Crop Stress Map (Predicted NDVI)
    cmap_stress = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    im2 = axes[1].imshow(health_map, cmap=cmap_stress, vmin=0, vmax=1)
    axes[1].set_title('Predicted Future Crop Stress Map (NDVI)')
    axes[1].set_xlabel('Patch Column')
    axes[1].set_ylabel('Patch Row')
    
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label('Predicted Mean NDVI (Higher is Healthier)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f'health_maps_{event_name}.png')
    plt.savefig(output_path)
    print(f"\nHealth maps saved successfully to: {output_path}")
    plt.show()
