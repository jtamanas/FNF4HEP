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
        super(BinaryFair, self).__init__()
        
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
        
        self.flow = Flow(data_dim=self.data_dim, 
                         context_dim=self.context_dim, 
                         hidden_dim=self.flow_hidden_dim, 
                         n_layers=self.flow_n_layers, 
                         transform_type=self.flow_transform_type)
        
        self.classifier = BinaryClassifier(self.embedding_dim)
        
    def forward(self, data, context, return_embedding=False):
        embedding, log_probs = self.flow._transform(data, context=context)
        label_pred = self.classifier(embedding)        
        if return_embedding:
            return label_pred, embedding
        return label_pred
    
    def sample(self, n_samples_per_context, context=None):
        return self.flow.sample(num_samples=n_samples_per_context, context=context)
        
    def _KL_loss(self, data, context, n_samples_per_context=128):        
        flipped_context = (~context.bool()).float()
        
        _, correct_log_probs = self.flow._transform(data, context)
        wrong_log_probs = self.flow.log_prob(data, flipped_context)
        
#         This is the way outlined in the paper, but I didn't get satisfactory results.
        KL = correct_log_probs - wrong_log_probs
        
#         Instead, I'll make both conditional distributions map on the unit normal
#         KL = zeros_log_probs - self.flow._distribution.log_prob(zeros_samples)
#         KL += ones_log_probs - self.flow._distribution.log_prob(ones_samples)
    
        return KL.mean()
    
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
    