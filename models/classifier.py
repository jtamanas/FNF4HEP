import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, data_dim, hidden_dim=32, n_layers=2, activation='GELU'):
        super(BinaryClassifier, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        
        layers = [nn.Linear(data_dim, hidden_dim), getattr(nn, activation)()]
        
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(getattr(nn, activation)())
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def loss(self, x, labels):
        y = self.forward(x)
        return self.criterion(y, labels)