"""
The main multi-modal sequence model. Fuses image features with tabular data
and performs multi-task learning for classification and health prediction.
"""
import torch
import torch.nn as nn

class MultiModalSeq2Seq(nn.Module):
    def __init__(self, cnn, num_tabular_features, num_classes, rnn_hidden_size=256, num_rnn_layers=2):
        super(MultiModalSeq2Seq, self).__init__()
        self.cnn = cnn
        cnn_feature_size = self.cnn.fc.out_features
        
        self.cnn_feature_bn = nn.BatchNorm1d(cnn_feature_size)
        
        combined_feature_size = cnn_feature_size + num_tabular_features
        self.encoder_rnn = nn.LSTM(combined_feature_size, rnn_hidden_size, 
                                   num_layers=num_rnn_layers, batch_first=True, 
                                   dropout=0.3 if num_rnn_layers > 1 else 0)
        
        # Head for disease classification
        self.fc_classify = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size // 2),
            nn.BatchNorm1d(rnn_hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(rnn_hidden_size // 2, num_classes)
        )
        
        # Head for predicting the single future health index value
        self.fc_health = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(rnn_hidden_size // 4, 1)
        )

    def forward(self, image_seq, tabular_seq):
        device = tabular_seq.device
        batch_size, timesteps, C, H, W = image_seq.shape
        
        cnn_features_list = []
        for t in range(timesteps):
            image_timestep = image_seq[:, t, :, :, :].to(device)
            cnn_out = self.cnn(image_timestep)
            cnn_features_list.append(cnn_out)
        
        cnn_features = torch.stack(cnn_features_list, dim=1)
        
        cnn_features_flat = cnn_features.view(-1, self.cnn.fc.out_features)
        cnn_features_bn = self.cnn_feature_bn(cnn_features_flat)
        cnn_features = cnn_features_bn.view(batch_size, timesteps, -1)

        combined_features = torch.cat([cnn_features, tabular_seq], dim=2)
        _, (hidden, cell) = self.encoder_rnn(combined_features)
        
        last_hidden_state = hidden[-1]
        
        classification_output = self.fc_classify(last_hidden_state)
        health_output = self.fc_health(last_hidden_state)
        
        return classification_output, health_output.squeeze(1)
