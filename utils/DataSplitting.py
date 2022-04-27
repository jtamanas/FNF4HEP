from models.galton import Galton
import torch
import numpy as np
from torch.utils.data import TensorDataset

params = {"batch_size": 32, "shuffle": True}

# Useful functions for generating data and generators

# todo: Want to be able to generate all high res
def gen_galton_data(
    alpha,
    Ntot_sims,
    n_high_steps=20,
    n_low_steps=5,
    n_balls=100,
    odds_skew=0.55,
    n_dims=1,
    sigma=0.4,
    all_high_res=False,
):
    """
    labels: 0 = skewed, 1 = unskewed
    context: 0 = low res, 1 = high res
    """

    Nsims_high_skew = int(round(Ntot_sims * alpha))
    Nsims_high_unskew = int(round(Ntot_sims * (1 - alpha)))

    g_high_skew = Galton(
        n_steps=n_high_steps,
        odds_right=odds_skew,
        n_dims=n_dims,
        noise=True,
        sigma=sigma,
    )
    g_high_unskew = Galton(
        n_steps=n_high_steps, odds_right=0.5, n_dims=n_dims, noise=True, sigma=sigma,
    )
    g_low_skew = Galton(
        n_steps=n_low_steps,
        odds_right=odds_skew,
        n_dims=n_dims,
        noise=True,
        sigma=sigma,
    )
    g_low_unskew = Galton(
        n_steps=n_low_steps, odds_right=0.5, n_dims=n_dims, noise=True, sigma=sigma
    )

    if all_high_res:
        pos_high_skew_arr = np.array(
            [g_high_skew.simulate(n_balls=n_balls) for _ in range(Ntot_sims)]
        )
        pos_high_unskew_arr = np.array(
            [g_high_unskew.simulate(n_balls=n_balls) for _ in range(Ntot_sims)]
        )
        pos_low_skew_arr = np.array([])
        pos_low_unskew_arr = np.array([])

    else:
        pos_high_skew_arr = np.array(
            [g_high_skew.simulate(n_balls=n_balls) for _ in range(Nsims_high_skew)]
        )
        pos_high_unskew_arr = np.array(
            [g_high_unskew.simulate(n_balls=n_balls) for _ in range(Nsims_high_unskew)]
        )
        pos_low_skew_arr = np.array(
            [g_low_skew.simulate(n_balls=n_balls) for _ in range(Nsims_high_unskew)]
        )
        pos_low_unskew_arr = np.array(
            [g_low_unskew.simulate(n_balls=n_balls) for _ in range(Nsims_high_skew)]
        )

    data = torch.cat(
        (
            torch.from_numpy(pos_high_skew_arr).float().reshape(-1, 1),
            torch.from_numpy(pos_low_skew_arr).float().reshape(-1, 1),
            torch.from_numpy(pos_low_unskew_arr).float().reshape(-1, 1),
            torch.from_numpy(pos_high_unskew_arr).float().reshape(-1, 1),
        )
    )

    labels = torch.cat(
        [
            torch.zeros(
                Nsims_high_skew * n_balls
            ),  # I don't think this works for n_dims != 1
            torch.zeros(Nsims_high_unskew * n_balls),
            torch.ones(Nsims_high_skew * n_balls),
            torch.ones(Nsims_high_unskew * n_balls),
        ]
    )

    context = torch.cat(
        [
            torch.ones(
                Nsims_high_skew * n_balls
            ),  # I don't think this works for n_dims != 1
            torch.zeros(Nsims_high_unskew * n_balls),
            torch.zeros(Nsims_high_skew * n_balls),
            torch.ones(Nsims_high_unskew * n_balls),
        ]
    )

    return data, labels, context


def split_data_by_context(
    data_train,
    labels_train,
    context_train,
    data_val,
    labels_val,
    context_val,
    data_test,
    labels_test,
    context_test,
):
    """
    Splits data into two groups, based on context (sensitive attribute) 
    """
    idx_0_train = context_train == 0
    idx_0_val = context_val == 0
    idx_0_test = context_test == 0

    data_0_train = data_train[idx_0_train]
    data_1_train = data_train[~idx_0_train]
    labels_0_train = labels_train[idx_0_train]
    labels_1_train = labels_train[~idx_0_train]
    context_0_train = context_train[idx_0_train]
    context_1_train = context_train[~idx_0_train]

    data_0_val = data_val[idx_0_val]
    data_1_val = data_val[~idx_0_val]
    labels_0_val = labels_val[idx_0_val]
    labels_1_val = labels_val[~idx_0_val]
    context_0_val = context_val[idx_0_val]
    context_1_val = context_val[~idx_0_val]

    data_0_test = data_test[idx_0_test]
    data_1_test = data_test[~idx_0_test]
    labels_0_test = labels_test[idx_0_test]
    labels_1_test = labels_test[~idx_0_test]
    context_0_test = context_test[idx_0_test]
    context_1_test = context_test[~idx_0_test]

    return (
        data_0_train,
        data_1_train,
        labels_0_train,
        labels_1_train,
        context_0_train,
        context_1_train,
        data_0_val,
        data_1_val,
        labels_0_val,
        labels_1_val,
        context_0_val,
        context_1_val,
        data_0_test,
        data_1_test,
        labels_0_test,
        labels_1_test,
        context_0_test,
        context_1_test,
    )


def get_fair_generators(
    data_train,
    labels_train,
    context_train,
    data_val,
    labels_val,
    context_val,
    data_test,
    labels_test,
    context_test,
):
    """
    returns testing and training generators for the fair normalizing flows
    """

    (
        data_0_train,
        data_1_train,
        labels_0_train,
        labels_1_train,
        context_0_train,
        context_1_train,
        data_0_val,
        data_1_val,
        labels_0_val,
        labels_1_val,
        context_0_val,
        context_1_val,
        data_0_test,
        data_1_test,
        labels_0_test,
        labels_1_test,
        context_0_test,
        context_1_test,
    ) = split_data_by_context(
        data_train,
        labels_train,
        context_train,
        data_val,
        labels_val,
        context_val,
        data_test,
        labels_test,
        context_test,
    )

    label_0_set = TensorDataset(data_0_train, labels_0_train, context_0_train)
    label_0_train_generator = torch.utils.data.DataLoader(label_0_set, **params)

    label_1_set = TensorDataset(data_1_train, labels_1_train, context_1_train)
    label_1_train_generator = torch.utils.data.DataLoader(label_1_set, **params)

    label_0_val_set = TensorDataset(data_0_val, labels_0_val, context_0_val)
    label_0_val_generator = torch.utils.data.DataLoader(label_0_val_set, **params)

    label_1_val_set = TensorDataset(data_1_val, labels_1_val, context_1_val)
    label_1_val_generator = torch.utils.data.DataLoader(label_1_val_set, **params)

    label_0_test_set = TensorDataset(
        data_0_test, labels_0_test, context_0_test
    )  # need same number of each, so use min length
    label_0_test_generator = torch.utils.data.DataLoader(
        label_0_test_set, batch_size=min(len(data_0_test), len(data_1_test))
    )

    label_1_test_set = TensorDataset(data_1_test, labels_1_test, context_1_test)
    label_1_test_generator = torch.utils.data.DataLoader(
        label_1_test_set, batch_size=min(len(data_0_test), len(data_1_test))
    )

    return (
        label_0_train_generator,
        label_1_train_generator,
        label_0_val_generator,
        label_1_val_generator,
        label_0_test_generator,
        label_1_test_generator,
    )


def get_embedding_generators(embedding_data, embedding_context, embedding_labels):
    """
    returns data generators with (embedded data, context) and (embedded data, labels) for
    adversarial and embedded label classifier respectively
    """
    embedding_context_set = TensorDataset(embedding_data, embedding_context)
    embedding_context_generator = torch.utils.data.DataLoader(
        embedding_context_set, **params
    )
    embedding_labels_set = TensorDataset(embedding_data, embedding_labels)
    embedding_labels_generator = torch.utils.data.DataLoader(
        embedding_labels_set, **params
    )

    return embedding_context_generator, embedding_labels_generator


def get_embedding_data(
    data_train, labels_train, context_train, data_test, labels_test, context_test, fair
):
    """
    returns data generators with (embedded data, context) and (embedded data, labels) for
    adversarial and embedded label classifier respectively. Also returns the raw embedded data, context, and labels
    for the adversarial and label classifiers. 
    """
    (
        data_0_train,
        data_1_train,
        labels_0_train,
        labels_1_train,
        context_0_train,
        context_1_train,
        data_0_test,
        data_1_test,
        labels_0_test,
        labels_1_test,
        context_0_test,
        context_1_test,
    ) = split_data_by_context(
        data_train, labels_train, context_train, data_test, labels_test, context_test
    )

    embedding_0_train, embedding_1_train = fair._embed(
        data_0_train,
        data_1_train,
        context_0_train.unsqueeze(1),
        context_1_train.unsqueeze(1),
    )
    embedding_0_test, embedding_1_test = fair._embed(
        data_0_test,
        data_1_test,
        context_0_test.unsqueeze(1),
        context_1_test.unsqueeze(1),
    )

    embedding_data_train = torch.cat(
        [embedding_0_train.detach(), embedding_1_train.detach()], dim=0
    )
    embedding_context_train = torch.cat([context_0_train, context_1_train], dim=0)
    embedding_labels_train = torch.cat([labels_0_train, labels_1_train], dim=0)

    embedding_data_test = torch.cat(
        [embedding_0_test.detach(), embedding_1_test.detach()], dim=0
    )
    embedding_context_test = torch.cat([context_0_test, context_1_test], dim=0)
    embedding_labels_test = torch.cat([labels_0_test, labels_1_test], dim=0)

    embedding_context_generator, embedding_labels_generator = get_embedding_generators(
        embedding_data_train, embedding_context_train, embedding_labels_train
    )

    return (
        embedding_context_generator,
        embedding_labels_generator,
        embedding_data_test,
        embedding_context_test,
        embedding_labels_test,
    )

