import torch
import torch.nn as nn
from tqdm.auto import trange


class BinaryClassifier(nn.Module):
    def __init__(self, data_dim, hidden_dim=32, n_layers=2, activation='ReLU'):
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
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        return self.model(x).squeeze()
    
    def loss(self, x, labels):
        y = self.forward(x).squeeze()
        return self.criterion(y, labels)

    def predict(self, x): #Don't think I need predict
        y = self.forward(x).squeeze()
        return torch.round(y)

    def fit(self, data_loader_train, n_steps = int(1e4), lr=1e-4, weight_decay=1e-4):

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)  

        self.train()
        classifier_loss = []
        print("Training Binary Label Classifier...")
        for step in trange(n_steps):
            data, labels = next(iter(data_loader_train)) #TODO add context to data_loader 
            
            optimizer.zero_grad()
            label_loss = self.loss(data, labels).mean()
            classifier_loss.append(label_loss.item())
            label_loss.backward()
            optimizer.step()

            #TODO: add validation set
        self.eval();
    
    def accuracy(self, data_test, labels_test):
        return ((self.forward(data_test) > 0.0) == labels_test).float().mean().item()
