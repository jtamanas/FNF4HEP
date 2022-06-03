from calendar import c
import torch
import torch.nn as nn
from .flow import Flow
from .classifier import BinaryClassifier
from sklearn.metrics import accuracy_score
from tqdm.auto import trange
import numpy as np
import copy
import math


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

        self.flow = Flow(
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


    def classifier_accuracy(self, embd, labels):
        label_pred = self.classifier(embd).sigmoid()
        bin_pred = (label_pred > 0.5).float()
        return accuracy_score(bin_pred, labels)

    def sample(self, context=None):
        return self.flow.sample(num_samples=context.shape[0], context=context)

    def _embed(self, data, context=None):
        return self.flow._transform(data, context=context)

    def optimal_adversary(
        self,
        data_0_loader,
        data_1_loader,
        probability_func=None,
    ):
        # TODO: In the continuous case, there should only be a single dataloader. 
        # Figure out how to handle this.
        
        data_0, _, context_0 = next(iter(data_0_loader))
        data_1, _, context_1 = next(iter(data_1_loader))

        if len(context_0.shape) == 1:
            context_0 = context_0.unsqueeze(1)
            context_1 = context_1.unsqueeze(1)

        z0, _, _ = self.flow._fair_forward(data_0, context_0)
        z1, _, _ = self.flow._fair_forward(data_1, context_1)

        logP_Z0_z0 = self._log_prob(z0, context_0, probability_func)
        logP_Z1_z0 = self._log_prob(z0, context_1, probability_func)
        logP_Z0_z1 = self._log_prob(z1, context_0, probability_func)
        logP_Z1_z1 = self._log_prob(z1, context_1, probability_func)

        mu_star_0 = logP_Z1_z0 >= logP_Z0_z0
        mu_star_1 = logP_Z1_z1 >= logP_Z0_z1

        mu_star_0_avg = (mu_star_0 * 1.0).mean()
        mu_star_1_avg = (mu_star_1 * 1.0).mean()

        stat_dist = mu_star_0_avg - mu_star_1_avg

        return stat_dist, mu_star_0, mu_star_1

    def _log_prob(self, z, context, probability_func):
        """
        This is the probabilty in the fair latent space
        Eqn. 4 of https://arxiv.org/abs/2106.05937
        """
        finv_z, logdet_finv_z = self.flow._transform.inverse(z, context)
        
        log_P_z = probability_func.log_prob(finv_z, context)

        logP_Z_z = log_P_z + logdet_finv_z
        
        return logP_Z_z

    def _reference_log_prob(self, z, context=None, probability_func=None):
        """
        Log prob of unit gaussian. 
        
        Constants are probably not needed but are nice to have because
        it enables to calculate exact KLs later on.
        """
        return - 0.5 * (z ** 2) - math.log(math.sqrt(2 * math.pi)) 
    

    def _KL_loss(self, data, context=None, probability_func=None):
        """
        data: torch.Tensor[batch_size, data_dim]
            Batch of features, does not include sensitive features
            
        context: torch.Tensor[batch_size, context_dim]
            Batch of sensitive features associated with data 
            
        probability_func: function
            Consumes data and context and returns a logprob
        
        
        This differs from the original fair normalizing flow paper in two ways
        
            1) The latent probabilities are mapped onto the unit gaussian rather than
                some arbitrary distribution that both pdfs converge on
                
            2) The distribution of contexts is not fixed, but  is assumed to be 
                uniformly sampled. 
                
        These two differences allow us to adapt fair normalizing flows to 
            continuous sensitive features.
        """
        z, _ = self.flow._transform(data, context)
        
        logP_Z_z = self._log_prob(z, context, probability_func)
        reference_log_prob = self._reference_log_prob(data, context, probability_func)

        L_0 = reference_log_prob - logP_Z_z

        return L_0.mean()

    def _classifier_loss(self, embedding, labels):
        """
        Not yet sure how to handle the context here.
        """
        label_preds = self.classifier.forward(embedding)
        return self.classifier.criterion(label_preds, labels)

    def loss(
        self,
        data,
        labels,
        context=None,
        return_all_losses=False,
        probability_func=None,
    ):
        embedding = self._embed(data, context)
        L_clf = self._classifier_loss(embedding, labels)
        
        if self.gamma == 0:
            return (
                torch.tensor(0),
                L_clf,
                L_clf,
            )

        L_KL = self._KL_loss(data, context, probability_func)

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
