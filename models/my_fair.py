import torch
import torch.nn as nn
from .flow import Flow
from .classifier import BinaryClassifier
from sklearn.metrics import accuracy_score


class BinaryFair(nn.Module):
    def __init__(
        self,
        data_dim,
        context_dim=None,
        flow_hidden_dim=32,
        flow_n_layers=1,
        flow_transform_type="MaskedAffineAutoregressiveTransform",
        classifier_hidden_dim=32,
        classifier_n_layers=2,
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

        self.flow1 = Flow(
            data_dim=self.data_dim,
            context_dim=self.context_dim,
            hidden_dim=self.flow_hidden_dim,
            n_layers=self.flow_n_layers,
            transform_type=self.flow_transform_type,
            num_bins=self.num_bins,
            tails=self.tails,
            tail_bound=self.tail_bound,
        )

        self.classifier = BinaryClassifier(self.embedding_dim) #Fair Classifier

    def forward(self, data, context):
        embedding0, embedding1 = self._embed(data_0=data, data_1=data)

        embeddings = embedding0 * (context == 0) + embedding1 * (context == 1)
        label_pred = self.classifier(embeddings).sigmoid()

        return label_pred

    def classifier_accuracy(self, embd, labels):
        label_pred = self.classifier(embd).sigmoid()
        bin_pred = (label_pred > 0.5).float()
        return accuracy_score(bin_pred, labels)

    def sample(self, context=None):
        samples_0 = self.flow0.sample(num_samples=context.shape[0])
        samples_1 = self.flow1.sample(num_samples=context.shape[0])

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
            embedding0, logabsdet0 = self.flow0._transform(data_0, context=None)

        if data_1 is not None:
            embedding1, logabsdet1 = self.flow1._transform(data_1, context=None)

        return embedding0, embedding1

    def _fair_embed(self, x0=None, x1=None, context_0=None, context_1=None):

        (
            z0,
            _,
        ) = self.flow0._transform(x0, None)
        (
            z1,
            _,
        ) = self.flow1._transform(x1, None)

        f0invz1, _ = self.flow0._transform.inverse(z1, None)
        f1invz0, _ = self.flow1._transform.inverse(z0, None)

        return f0invz1, f1invz0

    def optimal_adversary(
        self,
        data_0,
        data_1,
        context_0=None,
        context_1=None,
        probability_flow=None,
    ):
        z0, _, _ = self.flow0._fair_forward(data_0, None)
        z1, _, _ = self.flow1._fair_forward(data_1, None)

        logP_Z0_z0, logP_Z1_z0 = self._log_prob(
            z0, context_0, context_1, probability_flow
        )
        logP_Z0_z1, logP_Z1_z1 = self._log_prob(
            z1, context_0, context_1, probability_flow
        )

        mu_star_0 = logP_Z1_z0 >= logP_Z0_z0
        mu_star_1 = logP_Z1_z1 >= logP_Z0_z1

        mu_star_0_avg = (mu_star_0 * 1.0).mean()
        mu_star_1_avg = (mu_star_1 * 1.0).mean()

        stat_dist = mu_star_0_avg - mu_star_1_avg

        return stat_dist, mu_star_0, mu_star_1

    def _log_prob(self, z, context_0=None, context_1=None, probability_flow=None):

        f0invz, logdetf0invz = self.flow0._transform.inverse(z, None)
        f1invz, logdetf1invz = self.flow1._transform.inverse(z, None)

        log_P_0_z = probability_flow.log_prob(f0invz, context_0)
        log_P_1_z = probability_flow.log_prob(f1invz, context_1)

        logP_Z0_z = log_P_0_z + logdetf0invz
        logP_Z1_z = log_P_1_z + logdetf1invz

        return logP_Z0_z, logP_Z1_z

    def _KL_loss(
        self, data_0, data_1, context_0=None, context_1=None, probability_flow=None
    ):
        """
        data_0: [batch_size, data_dim]
            Data with label 0
        data_1: [batch_size, data_dim]
            Data with label 1

        Note in the paper, they have equal numbers of each class and then take
        the mean after adding. Here we'll take the mean first and then add scalars
        """
        z0, _, _ = self.flow0._fair_forward(data_0, None)
        z1, _, _ = self.flow1._fair_forward(data_1, None)

        logP_Z0_z0, logP_Z1_z0 = self._log_prob(
            z0, context_0, context_1, probability_flow
        )
        logP_Z0_z1, logP_Z1_z1 = self._log_prob(
            z1, context_0, context_1, probability_flow
        )

        L_0 = logP_Z0_z0 - logP_Z1_z0
        L_1 = logP_Z1_z1 - logP_Z0_z1

        return (L_0 + L_1).mean()

    def _classifier_loss(self, embd_0, embd_1, labels_0, labels_1):
        """
        Not yet sure how to handle the context here.
        """
        labels = torch.cat([labels_0, labels_1])

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
        probability_flow=None,
    ):
        if self.gamma == 0:
            embedding_0, embedding_1 = self._embed(data_0, data_1, context_0, context_1)
            return (
                torch.tensor(0),
                self._classifier_loss(embedding_0, embedding_1, labels_0, labels_1),
                self._classifier_loss(embedding_0, embedding_1, labels_0, labels_1),
            )

        L_KL = self._KL_loss(data_0, data_1, context_0, context_1, probability_flow)

        embedding_0, embedding_1 = self._embed(data_0, data_1, context_0, context_1)

        L_clf = self._classifier_loss(embedding_0, embedding_1, labels_0, labels_1)
        L_total = self.gamma * L_KL + (1.0 - self.gamma) * L_clf

        if return_all_losses:
            return L_KL, L_clf, L_total
        return L_total
