import torch
import torch.nn as nn
from .flow import Flow
from .classifier import BinaryClassifier


class BinaryFair(nn.Module):
    def __init__(self, 
                 data_dim, 
                 context_dim=1, 
                 flow_hidden_dim = 32,
                 flow_n_layers = 1,
                 flow_transform_type = 'MaskedAffineAutoregressiveTransform', 
                 classifier_hidden_dim = 32,
                 classifier_n_layers = 2,
                 classifier_activation = 'GELU',
                 gamma=0.5,
                ):
        super().__init__()
        
        self.data_dim = data_dim
        self.context_dim = None
        self.embedding_dim = data_dim  
        
        self.flow_hidden_dim = flow_hidden_dim 
        self.flow_n_layers = flow_n_layers
        self.flow_transform_type = flow_transform_type
        
        self.classifier_hidden_dim = classifier_hidden_dim
        self.classifier_n_layers = classifier_n_layers
        self.classifier_activation = classifier_activation
        
        self.gamma = gamma
        
        self.flow0 = Flow(data_dim=self.data_dim, 
                         context_dim=self.context_dim, 
                         hidden_dim=self.flow_hidden_dim, 
                         n_layers=self.flow_n_layers, 
                         transform_type=self.flow_transform_type)
        
        self.flow1 = Flow(data_dim=self.data_dim, 
                         context_dim=self.context_dim, 
                         hidden_dim=self.flow_hidden_dim, 
                         n_layers=self.flow_n_layers, 
                         transform_type=self.flow_transform_type)
        
        self.classifier = BinaryClassifier(self.embedding_dim)
        
    def forward(self, data, context, return_embedding=False):
        embedding, logabsdet = self.embed(data, context)
        
        label_pred = self.classifier(embedding)        
        if return_embedding:
            return label_pred, embedding
        return label_pred
    
    def sample(self, n_samples_per_context, context=None):
        samples_0 = self.flow0.sample(num_samples=n_samples_per_context)
        samples_1 = self.flow1.sample(num_samples=n_samples_per_context)
        if context is not None:
            context = context.bool()
            samples = samples_0*(~context) + samples_1*context
        else:
            samples = samples_0 + samples_1
        return samples
    
    def embed(self, data, context=None):
        # these flows are not made with contexts
        embedding0, logabsdet0 = self.flow0._transform(data, context=None)
        embedding1, logabsdet1 = self.flow1._transform(data, context=None)
        
        # fastest, cleanest way to selectively choose flow output. Only works for binarized context
        context = context.bool()
        embedding = embedding0*(~context) + embedding1*context
        logabsdet = logabsdet0*(~context) + logabsdet1*context
        return embedding, logabsdet
        
    def _KL_loss(self, data, context, n_samples_per_context=128):        
        """
        Note in the paper, they have equal numbers of each class and then take
        the mean after adding. Here we'll take the mean first and then add scalars
        """
        context = context.bool().flatten()
        
        # this is completely unbounded and idk how it worked for them in the paper
        KL = self.flow0.log_prob(data[~context]).mean()
        KL += -self.flow0._latent_log_prob(data[context]).mean()
        KL += self.flow1.log_prob(data[context]).mean()
        KL += -self.flow1._latent_log_prob(data[~context]).mean()

#         KL = self.flow0.log_prob(data[~context]).mean()
#         KL += self.flow1.log_prob(data[context]).mean()
        
        return -KL
    
    def _classifier_loss(self, data, context, labels):
        label_preds = self.forward(data, context)
        return self.classifier.criterion(label_preds, labels)
    
    def loss(self, data, context, labels, return_all_losses=False):
        batch_size = data.shape[0]
        
        L_KL = self._KL_loss(data, context)
        L_clf = self._classifier_loss(data, context, labels)
        L_total = self.gamma*L_KL + (1. - self.gamma)*L_clf
        
        if return_all_losses:
            return L_KL, L_clf, L_total
        return L_total
    