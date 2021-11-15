import torch
import torch.nn as nn
from .flow import Flow
from .classifier import BinaryClassifier


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
        classifier_activation="GELU",
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

        self.gamma = gamma

        self.flow0 = Flow(
            data_dim=self.data_dim,
            context_dim=self.context_dim,
            hidden_dim=self.flow_hidden_dim,
            n_layers=self.flow_n_layers,
            transform_type=self.flow_transform_type,
        )

        self.flow1 = Flow(
            data_dim=self.data_dim,
            context_dim=self.context_dim,
            hidden_dim=self.flow_hidden_dim,
            n_layers=self.flow_n_layers,
            transform_type=self.flow_transform_type,
        )

        self.classifier = BinaryClassifier(self.embedding_dim)

    def forward(self, data, context):
        embedding0, embedding1 = self._embed(data_0=data, data_1=data)

        embeddings = embedding0 * (context == 0) + embedding1 * (context == 1)
        label_pred = self.classifier(embeddings).sigmoid()

        return label_pred

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
            embedding0, logabsdet0 = self.flow0._transform(data_0, context=context_0)

        if data_1 is not None:
            embedding1, logabsdet1 = self.flow1._transform(data_1, context=context_1)

        return embedding0, embedding1

    def _fair_embed(self, x0=None, x1=None, context_0=None, context_1=None):

        (
            z0,
            _,
        ) = self.flow0._transform(x0, context_0)
        (
            z1,
            _,
        ) = self.flow1._transform(x1, context_1)

        f0invz1, logdetf0invz1 = self.flow0._transform.inverse(z1, context_0)
        f1invz0, logdetf1invz0 = self.flow1._transform.inverse(z0, context_1)

        return f0invz1, f1invz0

    def optimal_adversary(
        self,
        data_0,
        data_1,
        context_0=None,
        context_1=None,
        probability_flow=None,
    ):
        z0, _, _ = self.flow0._fair_forward(data_0, context_0)
        z1, _, _ = self.flow1._fair_forward(data_1, context_1)

        f0invz0, logdetf0invz0 = self.flow0._transform.inverse(z0, context_0)
        f1invz1, logdetf1invz1 = self.flow1._transform.inverse(z1, context_1)
        f0invz1, logdetf0invz1 = self.flow0._transform.inverse(z1, context_0)
        f1invz0, logdetf1invz0 = self.flow1._transform.inverse(z0, context_1)

        P_0_z0 = probability_flow.log_prob(f0invz0, context_0)
        P_1_z1 = probability_flow.log_prob(f1invz1, context_1)
        P_0_z1 = probability_flow.log_prob(f0invz1, context_0)
        P_1_z0 = probability_flow.log_prob(f1invz0, context_1)

        P_Z0_z0 = P_0_z0 + logdetf0invz0
        P_Z1_z1 = P_1_z1 + logdetf1invz1
        P_Z0_z1 = P_0_z1 + logdetf0invz1
        P_Z1_z0 = P_1_z0 + logdetf1invz0

        mu_star_0 = P_Z1_z0 >= P_Z0_z0
        mu_star_1 = P_Z1_z1 >= P_Z0_z1

        mu_star_0_avg = (mu_star_0 * 1.0).mean()
        mu_star_1_avg = (mu_star_1 * 1.0).mean()

        stat_dist = mu_star_0_avg - mu_star_1_avg

        return mu_star_0_avg, mu_star_1_avg, stat_dist

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
        z0, _, _ = self.flow0._fair_forward(data_0, context_0)
        z1, _, _ = self.flow1._fair_forward(data_1, context_1)

        f0invz0, logdetf0invz0 = self.flow0._transform.inverse(z0, context_0)
        f1invz1, logdetf1invz1 = self.flow1._transform.inverse(z1, context_1)
        f0invz1, logdetf0invz1 = self.flow0._transform.inverse(z1, context_0)
        f1invz0, logdetf1invz0 = self.flow1._transform.inverse(z0, context_1)

        log_P_0_z0 = probability_flow.log_prob(f0invz0, context_0)
        log_P_1_z1 = probability_flow.log_prob(f1invz1, context_1)
        log_P_0_z1 = probability_flow.log_prob(f0invz1, context_0)
        log_P_1_z0 = probability_flow.log_prob(f1invz0, context_1)

        logP_Z0_z0 = log_P_0_z0 + logdetf0invz0
        logP_Z1_z1 = log_P_1_z1 + logdetf1invz1
        logP_Z0_z1 = log_P_0_z1 + logdetf0invz1
        logP_Z1_z0 = log_P_1_z0 + logdetf1invz0

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
        L_KL = self._KL_loss(data_0, data_1, context_0, context_1, probability_flow)

        embedding_0, embedding_1 = self._embed(data_0, data_1, context_0, context_1)

        L_clf = self._classifier_loss(embedding_0, embedding_1, labels_0, labels_1)
        L_total = self.gamma * L_KL + (1.0 - self.gamma) * L_clf

        if return_all_losses:
            return L_KL, L_clf, L_total
        return L_total
