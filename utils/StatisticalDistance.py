import torch
from tqdm.auto import trange
from torch.utils.data import TensorDataset
from models.classifier import BinaryClassifier
from IPython import embed


def EmpiricalStatisticalDistance(
    samples_0,
    samples_1,
    hidden_dim=32,
    n_layers=2,
    activation="GELU",
    n_epochs=100,
    train_test_split=0.8,
    report_accuracy=True,
):

    discriminator = BinaryClassifier(
        samples_0.shape[-1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation,
    )

    data = torch.vstack([samples_0, samples_1])
    # this is really context, not labels
    labels = torch.vstack(
        [torch.zeros(samples_0.shape[0], 1), torch.ones(samples_1.shape[0], 1)]
    )

    training_idx = torch.rand_like(data[:, 0]) < train_test_split

    training_data = data[training_idx]
    training_labels = labels[training_idx]
    test_data = data[~training_idx]
    test_labels = labels[~training_idx]

    training_set = TensorDataset(training_data, training_labels)
    training_generator = torch.utils.data.DataLoader(
        training_set, batch_size=64, shuffle=True
    )

    optimizer = torch.optim.AdamW(discriminator.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    discriminator.train()
    for n_epoch in trange(n_epochs):
        for n_step, (data, labels) in enumerate(training_generator):
            optimizer.zero_grad()
            logits = discriminator(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    discriminator.eval()

    if report_accuracy:
        from sklearn.metrics import accuracy_score

        with torch.no_grad():
            cont_pred = discriminator(training_data).sigmoid()
            bin_pred = (cont_pred > 0.5).int()
            acc_train = accuracy_score(bin_pred, training_labels)
            print(f"Training set accuracy: {acc_train:.3f}")

            cont_pred = discriminator(test_data).sigmoid()
            bin_pred = (cont_pred > 0.5).int()
            acc_test = accuracy_score(bin_pred, test_labels)
            print(f"Test set accuracy: {acc_test:.3f}")

    test_preds = discriminator(test_data).sigmoid()
    # return torch.abs(
    #     test_preds[test_labels.bool()].mean() - test_preds[~test_labels.bool()].mean()
    # )
    return acc_test
