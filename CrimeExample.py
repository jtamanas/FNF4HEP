import torch
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

(
    data_train,
    data_test,
    context_train,
    context_test,
    labels_train,
    labels_test,
) = train_test_split(data, context, labels, test_size=0.2)


# Parameters
params = {"batch_size": 128, "shuffle": True}

# Split data by label
train_idx0 = context_train.flatten() == 0
test_idx0 = context_test.flatten() == 0

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

# Generators
label_0_set = TensorDataset(data_0_train, labels_0_train, context_0_train)
label_0_generator_train = torch.utils.data.DataLoader(label_0_set, **params)

label_1_set = TensorDataset(data_1_train, labels_1_train, context_1_train)
label_1_generator_train = torch.utils.data.DataLoader(label_1_set, **params)

label_0_set_test = TensorDataset(data_0_test, labels_0_test, context_0_test)
label_0_generator_test = torch.utils.data.DataLoader(label_0_set_test, **params)

label_1_set_test = TensorDataset(data_1_test, labels_1_test, context_1_test)
label_1_generator_test = torch.utils.data.DataLoader(label_1_set_test, **params)

data_set_train = TensorDataset(data_train, context_train)
data_generator_train = torch.utils.data.DataLoader(data_set_train, **params)

data_set_test = TensorDataset(data_train, context_train)
data_generator_test = torch.utils.data.DataLoader(data_set_test, **params)


# Create and train the flow that learns p0(x), p1(x)
probability_flow = Flow(
    data_dim=data_train.shape[-1],
    context_dim=context_train.shape[-1],
    n_layers=3,
    transform_type="MaskedAffineAutoregressiveTransform",
)
optimizer = torch.optim.AdamW(probability_flow.parameters(), lr=1e-3, weight_decay=1e-4)


n_steps_prob = 2000
probability_loss = []
probability_flow.train()
for n_step in trange(n_steps_prob):
    data, context = next(iter(data_generator_train))

    optimizer.zero_grad()

    loss = -probability_flow.log_prob(inputs=data, context=context).mean()
    probability_loss.append(loss.item())
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

    n_steps = 5000
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

        loss.backward()
        optimizer.step()
    Fair.eval()

    Fairs.append(Fair)


# calculate the statistical distance

data_0_test, labels_0_test, context_0_test = next(iter(label_0_generator_test))
data_1_test, labels_1_test, context_1_test = next(iter(label_1_generator_test))
stat_dists = []

for fair in Fairs:
    stat_dist, _, _ = fair.optimal_adversary(
        data_0_test,
        data_1_test,
        context_0=context_0_test,
        context_1=context_1_test,
        probability_flow=probability_flow,
    )
    stat_dists.append(abs(stat_dist.item()))


# Calculate the accuracy of label predictions

accs = []
for fair in Fairs:
    with torch.no_grad():
        idx = (data_file["context"] == 0).flatten()
        embedding_0, embedding_1 = fair._embed(
            data_file["data"][idx], data_file["data"][~idx]
        )

    acc_test = EmpiricalStatisticalDistance(
        embedding_0,
        embedding_1,
        hidden_dim=64,
        n_layers=5,
        n_epochs=100,
        report_accuracy=True,
    )
    accs.append(acc_test)

# plot result (can move to analysis script later)
plt.plot(stat_dists, accs, "-o", color="orange")
plt.xlim(0, 1)
plt.ylim(0.45, 1)
plt.xlabel("statistical distance")
plt.ylabel("adversarial accuracy?")
plt.savefig("Figures/fair_accuracy_vs_stat_dist.png")
