import torch
import copy
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from models.fair import BinaryFair
from models.flow import Flow
from models.classifier import BinaryClassifier
from utils.StatisticalDistance import EmpiricalStatisticalDistance
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import trange

# import data
data_file = torch.load("data/Crime/data.pt")
data = data_file["data"]
context = data_file["context"]
labels = data_file["labels"]

# split into train, val, test
(
    data_train,
    data_test,
    context_train,
    context_test,
    labels_train,
    labels_test,
) = train_test_split(data, context, labels, test_size=0.1)
(
    data_train,
    data_val,
    context_train,
    context_val,
    labels_train,
    labels_val,
) = train_test_split(data_train, context_train, labels_train, test_size=0.1)


# Parameters
params = {"batch_size": 64, "shuffle": True}

# Split data by label
# Todo: should compress this
train_idx0 = context_train.flatten() == 0
test_idx0 = context_test.flatten() == 0
val_idx0 = context_val.flatten() == 0

data_0_train = data_train[train_idx0]
labels_0_train = labels_train[train_idx0]
context_0_train = context_train[train_idx0]

data_1_train = data_train[~train_idx0]
labels_1_train = labels_train[~train_idx0]
context_1_train = context_train[~train_idx0]

data_0_test = data_test[test_idx0]
labels_0_test = labels_test[test_idx0]
context_0_test = context_test[test_idx0]

data_1_test = data_test[~test_idx0]
labels_1_test = labels_test[~test_idx0]
context_1_test = context_test[~test_idx0]

data_0_val = data_val[val_idx0]
labels_0_val = labels_val[val_idx0]
context_0_val = context_val[val_idx0]

data_1_val = data_val[~val_idx0]
labels_1_val = labels_val[~val_idx0]
context_1_val = context_val[~val_idx0]

# Generators
label_0_set = TensorDataset(data_0_train, labels_0_train, context_0_train)
label_0_generator_train = torch.utils.data.DataLoader(label_0_set, **params)

label_1_set = TensorDataset(data_1_train, labels_1_train, context_1_train)
label_1_generator_train = torch.utils.data.DataLoader(label_1_set, **params)

label_0_set_test = TensorDataset(data_0_test, labels_0_test, context_0_test)
label_0_generator_test = torch.utils.data.DataLoader(label_0_set_test, **params)

label_1_set_test = TensorDataset(data_1_test, labels_1_test, context_1_test)
label_1_generator_test = torch.utils.data.DataLoader(label_1_set_test, **params)

label_0_set_val = TensorDataset(data_0_val, labels_0_val, context_0_val)
label_0_generator_val = torch.utils.data.DataLoader(label_0_set_val, **params)

label_1_set_val = TensorDataset(data_1_val, labels_1_val, context_1_val)
label_1_generator_val = torch.utils.data.DataLoader(label_1_set_val, **params)

data_set_train = TensorDataset(data_train, context_train)
data_generator_train = torch.utils.data.DataLoader(data_set_train, **params)

data_set_test = TensorDataset(data_train, context_train)
data_generator_test = torch.utils.data.DataLoader(data_set_test, **params)

data_set_val = TensorDataset(data_val, context_val)
data_generator_val = torch.utils.data.DataLoader(data_set_val, **params)


# Create and train the flow that learns p0(x), p1(x)
probability_flow = Flow(
    data_dim=data_train.shape[-1],
    context_dim=context_train.shape[-1],
    n_layers=3,
    transform_type="MaskedAffineAutoregressiveTransform",
)
optimizer = torch.optim.AdamW(probability_flow.parameters(), lr=1e-3, weight_decay=1e-4)

n_steps_prob = 2000
num_epochs = 5
probability_loss = []
best_params = probability_flow.state_dict()
best_loss_val = np.inf

probability_flow.train()
for n_step in trange(n_steps_prob):
    data, context = next(iter(data_generator_train))

    optimizer.zero_grad()

    loss = -probability_flow.log_prob(inputs=data, context=context).mean()
    probability_loss.append(loss.item())

    if (n_step + 1) % (n_steps_prob / num_epochs) == 0:
        data_val, context_val = next(iter(data_generator_val))
        loss_val = -probability_flow.log_prob(
            inputs=data_val, context=context_val
        ).mean()
        if loss_val.item() < best_loss_val:
            best_loss_val = loss_val.item()
            best_params = copy.deepcopy(probability_flow.state_dict())

    loss.backward()
    optimizer.step()
probability_flow.eval()


# Create and train the fair flows
gammas = [0.0, 0.02, 0.1, 0.2, 0.9]
Fairs = []

for gamma in gammas:
    Fair = BinaryFair(
        data_dim=data_file["data"].shape[-1],
        context_dim=data_file["context"].shape[-1],
        flow_n_layers=4,
        flow_transform_type="MaskedAffineAutoregressiveTransform",
        classifier_hidden_dim=32,
        classifier_n_layers=4,
        classifier_activation="ReLU",
        gamma=gamma,
    )
    optimizer = torch.optim.AdamW(Fair.parameters(), lr=1e-3, weight_decay=1e-4)

    n_steps = 10000
    num_epochs = 5
    best_params_fair = Fair.state_dict()
    best_loss_val = np.inf

    Fair.train()
    for n_step in trange(n_steps):
        data_0, labels_0, context_0 = next(iter(label_0_generator_train))
        data_1, labels_1, context_1 = next(iter(label_1_generator_train))

        optimizer.zero_grad()

        _, _, loss = Fair.loss(
            data_0,
            data_1,
            labels_0=labels_0,
            labels_1=labels_1,
            context_0=context_0,
            context_1=context_1,
            return_all_losses=True,
            probability_flow=probability_flow,
        )

        if (n_step + 1) % (n_steps / num_epochs) == 0:
            data_0_val, labels_0_val, context_0_val = next(iter(label_0_generator_val))
            data_1_val, labels_1_val, context_1_val = next(iter(label_1_generator_val))
            _, _, loss_val = Fair.loss(
                data_0_val,
                data_1_val,
                labels_0=labels_0_val,
                labels_1=labels_1_val,
                context_0=context_0_val,
                context_1=context_1_val,
                return_all_losses=True,
                probability_flow=probability_flow,
            )
            if loss_val.item() < best_loss_val:
                best_loss_val = loss_val.item()
                best_params_fair = copy.deepcopy(Fair.state_dict())

        loss.backward()
        optimizer.step()
    Fair.eval()
    Fair.load_state_dict(best_params_fair)
    Fairs.append(Fair)


# Save models
torch.save(probability_flow.state_dict(), "saved_models/probability_flow.pth")

for fair in Fairs:
    torch.save(fair.state_dict(), "saved_models/fair_" + str(fair.gamma) + ".pth")

np.savez(
    "saved_data/data.npz",
    data_test=data_test,
    labels_test=labels_test,
    context_test=context_test,
)
