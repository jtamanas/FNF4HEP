import torch
import torch.nn as nn
from tqdm.auto import trange
import numpy as np
import copy


class BinaryClassifier(nn.Module):
    def __init__(self, data_dim, hidden_dim=32, n_layers=2, activation="ReLU"):
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

    def predict(self, x):  # Don't think I need predict
        y = self.forward(x).squeeze()
        return torch.round(y)

    def fit(
        self,
        data_loader_train,
        data_loader_val,
        lr=1e-4,
        weight_decay=1e-4,
        max_num_epochs=100,
        n_steps_per_epoch=int(5e2),
        patience=10,
    ):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_loss_val = np.inf
        patience_count = 0
        val_loss = []

        self.train()
        for step in trange(n_steps_per_epoch * max_num_epochs):
            data, labels = next(iter(data_loader_train))

            optimizer.zero_grad()
            label_loss = self.loss(data, labels).mean()

            if (step + 1) % (n_steps_per_epoch) == 0:
                data_val, labels_val = next(iter(data_loader_val))
                loss_val = self.loss(data_val, labels_val).mean()
                val_loss.append(loss_val.item())

                if loss_val.item() < best_loss_val:
                    best_loss_val = loss_val.item()
                    best_params = copy.deepcopy(self.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1

                if patience_count >= patience:
                    print(
                        "Early stopping reached after {} epochs".format(
                            int((step + 1) / n_steps_per_epoch)
                        )
                    )
                    break

            label_loss.backward()
            optimizer.step()
        if step == n_steps_per_epoch * max_num_epochs - 1:
            print(
                "Training finished after {} epochs (max_num_epochs)".format(
                    max_num_epochs
                )
            )
        self.eval()
        self.load_state_dict(best_params)

    def accuracy(self, data_test, labels_test):
        return ((self.forward(data_test) > 0.0) == labels_test).float().mean().item()
