import torch
import torch.nn as nn
from .flow import Flow
from .classifier import BinaryClassifier


class ProbabilityFlow(nn.Module):
    def __init__(
        self,
        data_dim,
        context_dim=None,
        flow_hidden_dim=32,
        flow_n_layers=1,
        flow_transform_type="MaskedAffineAutoregressiveTransform"
    ):
        super().__init__()

        self.data_dim = data_dim
        self.context_dim = context_dim
        self.embedding_dim = data_dim

        self.flow_hidden_dim = flow_hidden_dim
        self.flow_n_layers = flow_n_layers
        self.flow_transform_type = flow_transform_type

        self.flow = Flow(
            data_dim=self.data_dim,
            context_dim=self.context_dim,
            hidden_dim=self.flow_hidden_dim,
            n_layers=self.flow_n_layers,
            transform_type=self.flow_transform_type,
        )

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


    def loss(
        self,
        data_0,
        data_1,
        labels_0,
        labels_1,
        context_0=None,
        context_1=None,
        return_all_losses=False,
    ):
        L_KL = self._KL_loss(data_0, data_1, context_0, context_1)

        embedding_0, embedding_1 = self._embed(data_0, data_1, context_0, context_1)

        L_clf = self._classifier_loss(embedding_0, embedding_1, labels_0, labels_1)
        L_total = self.gamma * L_KL + (1.0 - self.gamma) * L_clf

        # if return_all_losses:
        #     return L_KL, L_clf, L_total
        return L_total
