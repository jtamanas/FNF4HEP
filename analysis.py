import numpy as np
import matplotlib.pyplot as plt
import torch
from models.flow import Flow
from models.fair import BinaryFair
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset
from utils.StatisticalDistance import EmpiricalStatisticalDistance

# this is only the test data set
datadir = "saved_data/"
data_file = np.load(datadir + "data.npz")
data_test = torch.Tensor(data_file["data_test"])
context_test = torch.Tensor(data_file["context_test"])
labels_test = torch.Tensor(data_file["labels_test"])

modeldir = "saved_models/"

probability_flow = Flow(
    data_dim=data_test.shape[-1],
    context_dim=context_test.shape[-1],
    n_layers=3,
    transform_type="MaskedAffineAutoregressiveTransform",
)
probability_flow.load_state_dict(torch.load(modeldir + "probability_flow.pth"))
probability_flow.eval()

gammas = [0.0, 0.02, 0.1, 0.2, 0.9]
Fairs = []

for gamma in gammas:
    Fair = BinaryFair(
        data_dim=data_test.shape[-1],
        context_dim=context_test.shape[-1],
        flow_n_layers=4,
        flow_transform_type="MaskedAffineAutoregressiveTransform",
        classifier_hidden_dim=32,
        classifier_n_layers=4,
        classifier_activation="ReLU",
        gamma=gamma,
    )
    Fair.load_state_dict(torch.load(modeldir + "fair_{}.pth".format(gamma)))
    Fair.eval()
    Fairs.append(Fair)

# Parameters

# Generators
# data_set_test = TensorDataset(data_test, context_test)
# data_generator_test = torch.utils.data.DataLoader(data_set_test, **params)

# data, context = next(iter(data_generator_test))
# data_embedded = TSNE(n_components=2).fit_transform(data)
# samples = probability_flow._sample(num_samples=1, context=context)

# catData = torch.cat((data, samples.squeeze(dim=1)))
# catData_embedded = TSNE(n_components=2).fit_transform(catData.detach().numpy())
# catLabel = torch.cat((torch.zeros(data.shape[0]), torch.ones(samples.shape[0])))

# plt.scatter(*catData_embedded.T, c=catLabel, alpha=0.1, cmap="coolwarm")
# plt.show()


# make analysis plots

# split data by label
data_0_test = data_test[context_test.flatten() == 0]
labels_0_test = labels_test[context_test.flatten() == 0]
context_0_test = context_test[context_test.flatten() == 0]

data_1_test = data_test[context_test.flatten() == 1]
labels_1_test = labels_test[context_test.flatten() == 1]
context_1_test = context_test[context_test.flatten() == 1]

# Losing some data here, but want batches to have same length
params = {
    "batch_size": min(data_1_test.shape[0], data_0_test.shape[0]),
    "shuffle": True,
}

label_0_set_test = TensorDataset(data_0_test, labels_0_test, context_0_test)
label_0_generator_test = torch.utils.data.DataLoader(label_0_set_test, **params)

label_1_set_test = TensorDataset(data_1_test, labels_1_test, context_1_test)
label_1_generator_test = torch.utils.data.DataLoader(label_1_set_test, **params)

data_0_test, labels_0_test, context_0_test = next(iter(label_0_generator_test))
data_1_test, labels_1_test, context_1_test = next(iter(label_1_generator_test))

# calculate statistical distance
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

add_accs = []
for fair in Fairs:
    with torch.no_grad():
        # idx = (data_file["context"] == 0).flatten()
        embedding_0, embedding_1 = fair._embed(data_0_test, data_1_test)

    # I don't think this is the fair classifier
    # Adversarial accuracy
    add_acc = EmpiricalStatisticalDistance(
        embedding_0,
        embedding_1,
        hidden_dim=32,
        n_layers=3,
        n_epochs=100,
        report_accuracy=True,
    )
    add_accs.append(add_acc)

# plot result (can move to analysis script later)
plt.plot(stat_dists, add_accs, "-o", color="orange")
plt.xlim(0, 1)
plt.ylim(0.4, 1)
plt.xlabel("Statistical distance")
plt.ylabel("Adversarial Accuracy")
plt.savefig("Figures/add_accuracy_vs_stat_dist.png")
plt.close()

bin_accs = []
for fair in Fairs:
    with torch.no_grad():
        # idx = (data_file["context"] == 0).flatten()
        embedding_0, embedding_1 = fair._embed(data_0_test, data_1_test)

        embedding = torch.cat((embedding_0, embedding_1))
        labels = torch.cat((labels_0_test, labels_1_test))
    # I don't think this is the fair classifier
    # Adversarial accuracy
    bin_acc = fair.classifier_accuracy(embedding, labels)
    bin_accs.append(bin_acc)

# plot result (can move to analysis script later)
plt.plot(stat_dists, bin_accs, "-o", color="orange")
plt.xlim(0, 1)
plt.ylim(0.4, 1)
plt.xlabel("Statistical distance")
plt.ylabel("Binary Classifier Accuracy")
plt.savefig("Figures/bin_accuracy_vs_stat_dist.png")
