"""
PyTorch Dataset classes for loading and preparing the preprocessed
MATLAB-enhanced .mat files.
"""
import os
import numpy as np
import scipy.io
from torch.utils.data import Dataset
from src.config import globals as config

class LocalSequenceDataset(Dataset):
    """Dataset for training and validation."""
    def __init__(self, data_dir, event_metadata, iot_data, scalers, encoders):
        self.iot_data = iot_data
        self.scalers = scalers
        self.encoders = encoders
        self.n_steps_in = config.N_STEPS_IN
        self.total_timesteps = config.N_STEPS_IN + config.N_STEPS_OUT
        self.samples = self._create_samples(data_dir, event_metadata)

    def _create_samples(self, data_dir, event_metadata):
        samples = []
        for event_name in event_metadata.keys():
            event_path = os.path.join(data_dir, event_name)
            if not os.path.isdir(event_path): continue
            
            mat_files = sorted([os.path.join(event_path, f) for f in os.listdir(event_path) if f.endswith('.mat')])
            if len(mat_files) < self.total_timesteps: continue
            
            try:
                first_mat = scipy.io.loadmat(mat_files[0])
                num_patches = first_mat['patches'].shape[0]
                for patch_idx in range(num_patches):
                    samples.append({'event': event_name, 'patch_idx': patch_idx, 'mat_files': mat_files})
            except Exception as e:
                print(f"Warning: Could not process {event_name}. Error: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        event_name = sample_info['event']
        patch_idx = sample_info['patch_idx']
        mat_files = sample_info['mat_files']

        img_sequence_list = [scipy.io.loadmat(f)['patches'][patch_idx] for f in mat_files[:self.total_timesteps]]
        img_sequence = np.array(img_sequence_list, dtype=np.float32)

        # NDVI is the 5th channel (index 4) in the 10-channel data
        future_ndvi_mean = img_sequence[self.n_steps_in:, :, :, 4].mean()

        meta = config.EVENT_METADATA[event_name]
        iot_values = self.iot_data[event_name][:self.total_timesteps]
        iot_normalized = self.scalers['iot'].transform(iot_values)
        crop_encoded = self.encoders['crop'].transform([[meta['crop_type']]])[0]
        disease_encoded = self.encoders['disease'].transform([[meta['disease']]])[0]
        
        tabular_features = [np.concatenate([iot_normalized[i], crop_encoded, disease_encoded]) for i in range(self.n_steps_in)]
        
        X_img = torch.from_numpy(img_sequence[:self.n_steps_in]).permute(0, 3, 1, 2)
        X_tabular = torch.from_numpy(np.array(tabular_features, dtype=np.float32))
        y_class = meta['label']
        y_health = np.float32(future_ndvi_mean)
        
        return (X_img, X_tabular), (y_class, y_health)

class InferenceDataset(Dataset):
    """
    Dataset for inference. Loads all patches for a single specified event
    and provides only the input data (X).
    """
    def __init__(self, data_dir, event_name, iot_data, scalers, encoders):
        self.iot_data = iot_data
        self.scalers = scalers
        self.encoders = encoders
        self.n_steps_in = config.N_STEPS_IN
        self.event_name = event_name

        event_path = os.path.join(data_dir, event_name)
        self.mat_files = sorted([os.path.join(event_path, f) for f in os.listdir(event_path) if f.endswith('.mat')])
        
        self.num_patches = 0
        if self.mat_files:
            try:
                first_mat = scipy.io.loadmat(self.mat_files[0])
                self.num_patches = first_mat['patches'].shape[0]
            except Exception as e:
                print(f"Error loading {self.mat_files[0]} to determine patch count: {e}")

    def __len__(self):
        return self.num_patches

    def __getitem__(self, patch_idx):
        img_sequence_list = [scipy.io.loadmat(f)['patches'][patch_idx] for f in self.mat_files[:self.n_steps_in]]
        img_sequence = np.array(img_sequence_list, dtype=np.float32)
        
        meta = config.EVENT_METADATA[self.event_name]
        iot_values = self.iot_data[self.event_name][:self.n_steps_in]
        iot_normalized = self.scalers['iot'].transform(iot_values)
        crop_encoded = self.encoders['crop'].transform([[meta['crop_type']]])[0]
        disease_encoded = self.encoders['disease'].transform([[meta['disease']]])[0]
        
        tabular_features = [np.concatenate([iot_normalized[i], crop_encoded, disease_encoded]) for i in range(self.n_steps_in)]
        
        X_img = torch.from_numpy(img_sequence).permute(0, 3, 1, 2)
        X_tabular = torch.from_numpy(np.array(tabular_features, dtype=np.float32))
        
        return (X_img, X_tabular)


