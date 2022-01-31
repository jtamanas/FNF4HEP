# Define flow
from nflows import transforms, distributions, flows


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
        tail_bound=1.0
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
