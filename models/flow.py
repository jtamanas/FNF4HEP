# Define flow
from nflows import transforms, distributions, flows
from .transforms import MaskedUMNNAutoregressiveTransform
from .classifier import BinaryClassifier
from tqdm.auto import trange
import torch
import copy
import numpy as np


class Flow(flows.Flow):
    def __init__(
        self,
        data_dim,
        context_dim=0,
        hidden_dim=32,
        n_layers=1,
        transform_type="MaskedAffineAutoregressiveTransform",
        num_bins=0,
        tails=None,
        tail_bound=1.0,
        gamma=0.5,
    ):

        self.data_dim = data_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.transform_type = transform_type
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound

        # Define an invertible transformation.
        transform = transforms.CompositeTransform(
            [self.create_transform(self.transform_type) for _ in range(self.n_layers)]
        )
        base_distribution = distributions.StandardNormal(shape=[data_dim])

        super().__init__(transform=transform, distribution=base_distribution)

        self.classifier = BinaryClassifier(data_dim=data_dim)
        self.gamma = gamma

    def create_transform(self, transform_type):
        linear = transforms.ReversePermutation(features=self.data_dim)

        if transform_type == "MaskedAffineAutoregressiveTransform":
            base = transforms.MaskedAffineAutoregressiveTransform(
                features=self.data_dim,
                context_features=self.context_dim,
                hidden_features=self.hidden_dim,
            )
        elif transform_type == "MaskedPiecewiseQuadraticAutoregressiveTransform":
            base = transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
                features=self.data_dim,
                context_features=self.context_dim,
                hidden_features=self.hidden_dim,
                num_bins=self.num_bins,
                tails=self.tails,
                tail_bound=self.tail_bound,
            )
        elif transform_type == "MaskedUMNNAutoregressiveTransform":
            base = MaskedUMNNAutoregressiveTransform(
                features=self.data_dim,
                context_features=self.context_dim,
                hidden_features=self.hidden_dim,
            )
        return transforms.CompositeTransform([linear, base])

    def _latent_log_prob(self, latents, context=None):
        embedded_context = self._embedding_net(context)
        log_prob = self._distribution.log_prob(latents, context=embedded_context)
        return log_prob

    def _fair_forward(self, data, context=None):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(data, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        # data logprob is log_prob + logabsdet
        return noise, log_prob, logabsdet

    def prob_flow_loss(self, data, labels, context):

        # Make sure labels and context are is correct shape
        if len(context.shape) == 1:
            context = context.unsqueeze(1)
        if len(labels.shape) != 1:
            labels = labels.squeeze()

        mle_loss = -self.log_prob(inputs=data, context=context).mean()
        label_loss = self.classifier.loss(data, labels).mean()

        return self.gamma * mle_loss + (1 - self.gamma) * label_loss

    def fit_prob_flow(
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
        for n_step in trange(n_steps_per_epoch * max_num_epochs):
            data, labels, context = next(iter(data_loader_train))

            optimizer.zero_grad()
            loss = self.prob_flow_loss(data, labels, context)

            if (n_step + 1) % (n_steps_per_epoch) == 0:
                data_val, labels_val, context_val = next(iter(data_loader_val))
                loss_val = self.prob_flow_loss(data_val, labels_val, context_val)
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
                            int((n_step + 1) / n_steps_per_epoch)
                        )
                    )
                    break

            loss.backward()
            optimizer.step()
        self.eval()
        self.load_state_dict(best_params)

        return val_loss

