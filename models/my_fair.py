import torch
import torch.nn as nn
from .flow import Flow
from .classifier import BinaryClassifier
from sklearn.metrics import accuracy_score
from tqdm.auto import trange
import numpy as np
import copy


class BinaryFair(nn.Module):
    def __init__(
        self,
        data_dim,
        context_dim=None,
        flow_hidden_dim=32,
        flow_n_layers=4,
        flow_transform_type="MaskedAffineAutoregressiveTransform",
        classifier_hidden_dim=32,
        classifier_n_layers=4,
        classifier_activation="ReLU",
        num_bins=0,
        tails=None,
        tail_bound=1.0,
        gamma=0.5,
    ):
        super().__init__()

        self.data_dim = data_dim
        self.context_dim = context_dim
        self.embedding_dim = data_dim

        self.flow_hidden_dim = flow_hidden_dim
        self.flow_n_layers = flow_n_layers
        self.flow_transform_type = flow_transform_type

        self.classifier_hidden_dim = classifier_hidden_dim
        self.classifier_n_layers = classifier_n_layers
        self.classifier_activation = classifier_activation

        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound

        self.gamma = gamma

        self.flow0 = Flow(
            data_dim=self.data_dim,
            context_dim=self.context_dim,
            hidden_dim=self.flow_hidden_dim,
            n_layers=self.flow_n_layers,
            transform_type=self.flow_transform_type,
            num_bins=self.num_bins,
            tails=self.tails,
            tail_bound=self.tail_bound,
        )

        self.classifier = BinaryClassifier(self.embedding_dim)  # Fair Classifier

    # def forward(self, data, context):
    #     embedding0, embedding1 = self._embed(data_0=data, data_1=data)

    #     embeddings = embedding0 * (context == 0) + embedding1 * (context == 1)
    #     label_pred = self.classifier(embeddings).sigmoid()

    #     return label_pred

    def classifier_accuracy(self, embd, labels):
        label_pred = self.classifier(embd).sigmoid()
        bin_pred = (label_pred > 0.5).float()
        return accuracy_score(bin_pred, labels)

    def sample(self, context_0=None, context_1=None):
        samples_0 = self.flow0.sample(num_samples=context.shape[0], context=context_0)
        samples_1 = self.flow0.sample(num_samples=context.shape[0], context=context_1)

        if context is not None:
            context = context.bool()
            samples = samples_0 * (~context) + samples_1 * context
        else:
            samples = samples_0 + samples_1
        return samples

    def _embed(self, data_0=None, data_1=None, context_0=None, context_1=None):
        embedding0 = None
        embedding1 = None

        if data_0 is not None:
            # embedding0, logabsdet0, _ = self.flow0._fair_forward(
            #     data_0, context=context_0
            # )
            embedding0, logabsdet0 = self.flow0._transform(data_0, context=context_0)
            # ? should these be _fair_forward or just _transform?

        # if data_1 is not None:
        #     embedding1, logabsdet1, _ = self.flow0._fair_forward(
        #         data_1, context=context_1
        #     )
        if data_1 is not None:
            embedding1, logabsdet1 = self.flow0._transform(data_1, context=context_1)

        return embedding0, embedding1

    # def _fair_embed(self, x0=None, x1=None, context_0=None, context_1=None):

    #     (z0, _,) = self.flow0._transform(x0, context_0)
    #     (z1, _,) = self.flow0._transform(x1, context_1)

    #     f0invz1, _ = self.flow0._transform.inverse(z1, context_0)
    #     f1invz0, _ = self.flow0._transform.inverse(z0, context_1)

    #     return f0invz1, f1invz0

    def optimal_adversary(
        self, data_0_loader, data_1_loader, probability_func=None,
    ):

        data_0, _, context_0 = next(iter(data_0_loader))
        data_1, _, context_1 = next(iter(data_1_loader))

        if len(context_0.shape) == 1:
            context_0 = context_0.unsqueeze(1)
            context_1 = context_1.unsqueeze(1)

        z0, _, _ = self.flow0._fair_forward(data_0, context_0)
        z1, _, _ = self.flow0._fair_forward(data_1, context_1)

        logP_Z0_z0, logP_Z1_z0 = self._log_prob(
            z0, context_0, context_1, probability_func
        )
        logP_Z0_z1, logP_Z1_z1 = self._log_prob(
            z1, context_0, context_1, probability_func
        )

        mu_star_0 = logP_Z1_z0 >= logP_Z0_z0
        mu_star_1 = logP_Z1_z1 >= logP_Z0_z1

        mu_star_0_avg = (mu_star_0 * 1.0).mean()
        mu_star_1_avg = (mu_star_1 * 1.0).mean()

        stat_dist = mu_star_0_avg - mu_star_1_avg

        return stat_dist, mu_star_0, mu_star_1

    def _log_prob(self, z, context_0=None, context_1=None, probability_func=None):

        f0invz, logdetf0invz = self.flow0._transform.inverse(z, context_0)
        f1invz, logdetf1invz = self.flow0._transform.inverse(z, context_1)

        log_P_0_z = probability_func.log_prob(f0invz, context_0)
        log_P_1_z = probability_func.log_prob(f1invz, context_1)

        logP_Z0_z = log_P_0_z + logdetf0invz
        logP_Z1_z = log_P_1_z + logdetf1invz

        return logP_Z0_z, logP_Z1_z

        # probability flow loss + classifier loss

    def _KL_loss(
        self, data_0, data_1, context_0=None, context_1=None, probability_func=None
    ):
        """
        data_0: [batch_size, data_dim]
            Data with label 0
        data_1: [batch_size, data_dim]
            Data with label 1

        Note in the paper, they have equal numbers of each class and then take
        the mean after adding. Here we'll take the mean first and then add scalars
        """
        # z0, _, _ = self.flow0._fair_forward(data_0, context_0)
        # z1, _, _ = self.flow0._fair_forward(data_1, context_1)

        # ? Do I need to use fair forward which uses the embedding net? Or does
        # ? transform do this as well?
        # I think just _transform is fine because the embedding net is just
        # identity by default. Maybe for forward compatability tho?

        z0, _ = self.flow0._transform(data_0, context_0)
        z1, _ = self.flow0._transform(data_1, context_1)

        logP_Z0_z0, logP_Z1_z0 = self._log_prob(
            z0, context_0, context_1, probability_func
        )
        logP_Z0_z1, logP_Z1_z1 = self._log_prob(
            z1, context_0, context_1, probability_func
        )

        L_0 = logP_Z0_z0 - logP_Z1_z0
        L_1 = logP_Z1_z1 - logP_Z0_z1

        # self._log_prob(z, context, probability_func)

        return (L_0 + L_1).mean()

    def _classifier_loss(self, embd_0, embd_1, labels_0, labels_1):
        """
        Not yet sure how to handle the context here.
        """
        labels = torch.cat([labels_0, labels_1]).squeeze()

        embds = torch.cat([embd_0, embd_1])
        label_preds = self.classifier.forward(embds)

        return self.classifier.criterion(label_preds, labels)

    def loss(
        self,
        data_0,
        data_1,
        labels_0,
        labels_1,
        context_0=None,
        context_1=None,
        return_all_losses=False,
        probability_func=None,
    ):
        if self.gamma == 0:
            embedding_0, embedding_1 = self._embed(data_0, data_1, context_0, context_1)
            return (
                torch.tensor(0),
                self._classifier_loss(embedding_0, embedding_1, labels_0, labels_1),
                self._classifier_loss(embedding_0, embedding_1, labels_0, labels_1),
            )

        L_KL = self._KL_loss(data_0, data_1, context_0, context_1, probability_func)

        embedding_0, embedding_1 = self._embed(data_0, data_1, context_0, context_1)

        L_clf = self._classifier_loss(embedding_0, embedding_1, labels_0, labels_1)
        L_total = self.gamma * L_KL + (1.0 - self.gamma) * L_clf

        if return_all_losses:
            return L_KL, L_clf, L_total
        return L_total

    def fit(
        self,
        data_0_loader_train,
        data_1_loader_train,
        data_0_loader_val,
        data_1_loader_val,
        probability_func,
        lr=1e-3,
        weight_decay=1e-4,
        max_num_epochs=100,
        n_steps_per_epoch=int(5e2),
        patience=10,
    ):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        best_loss_val = np.inf
        patience_count = 0
        val_loss = []

        self.train()
        fair_classifier_loss = []
        for step in trange(n_steps_per_epoch * max_num_epochs):
            data_0, labels_0, context_0 = next(iter(data_0_loader_train))
            data_1, labels_1, context_1 = next(iter(data_1_loader_train))

            if len(context_0.shape) == 1:
                context_0 = context_0.unsqueeze(1)
                context_1 = context_1.unsqueeze(1)

            optimizer.zero_grad()

            _, _, loss = self.loss(
                data_0,
                data_1,
                labels_0=labels_0,
                labels_1=labels_1,
                context_0=context_0,
                context_1=context_1,
                return_all_losses=True,
                probability_func=probability_func,
            )

            if (step + 1) % (n_steps_per_epoch) == 0:
                data_0_val, labels_0_val, context_0_val = next(iter(data_0_loader_val))
                data_1_val, labels_1_val, context_1_val = next(iter(data_1_loader_val))

                if len(context_0_val.shape) == 1:
                    context_0_val = context_0_val.unsqueeze(1)
                    context_1_val = context_1_val.unsqueeze(1)

                _, _, loss_val = self.loss(
                    data_0_val,
                    data_1_val,
                    labels_0=labels_0_val,
                    labels_1=labels_1_val,
                    context_0=context_0_val,
                    context_1=context_1_val,
                    return_all_losses=True,
                    probability_func=probability_func,
                )
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

            loss.backward()
            if np.isnan(loss.item()):
                print("Loss is nan")
                break
            optimizer.step()
        self.eval()

        return val_loss

    def adversarial_accuracy(self, data_0_loader, data_1_loader, adv_classifier):

        data_0, _, context_0 = next(iter(data_0_loader))
        data_1, _, context_1 = next(iter(data_1_loader))

        embedding_0, embedding_1 = self._embed(data_0, data_1)

        embedded_data_test = torch.cat([embedding_0, embedding_1], dim=0)
        embedded_context_test = torch.cat([context_0, context_1], dim=0)

        acc = (
            (
                (adv_classifier.forward(embedded_data_test) > 0.0)
                == embedded_context_test
            )
            .float()
            .mean()
            .item()
        )

        return acc
