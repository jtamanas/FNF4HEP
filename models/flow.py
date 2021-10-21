# Define flow
from nflows import transforms, distributions, flows


class Flow(flows.Flow):
    def __init__(self, data_dim, context_dim=0, hidden_dim=32, n_layers=1, transform_type='MaskedAffineAutoregressiveTransform'):
        self.data_dim = data_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.transform_type = transform_type
        
        # Define an invertible transformation.
        transform = transforms.CompositeTransform([self.create_transform(self.transform_type) for _ in range(self.n_layers)])
        base_distribution = distributions.StandardNormal(shape=[data_dim])
        
        super().__init__(transform=transform, distribution=base_distribution)
    
    def create_transform(self, transform_type):
        linear = transforms.ReversePermutation(features=self.data_dim)

        if transform_type == 'MaskedAffineAutoregressiveTransform':
            base = transforms.MaskedAffineAutoregressiveTransform(features=self.data_dim, 
                                                                  context_features=self.context_dim, 
                                                                  hidden_features=self.hidden_dim)
        return transforms.CompositeTransform([linear, base])
    
    def _latent_log_prob(self, latents, context=None):
        """
        This is dumb and wasteful, but I think it works to get the "off diagonal" KL terms
        """
        embedded_context = self._embedding_net(context)
        data, inv_logabsdet = self._transform.inverse(latents, context=embedded_context)
        return self.log_prob(data, context=context)