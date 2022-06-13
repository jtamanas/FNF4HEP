"""
Module for constructing flows.
"""


from typing import Callable, NamedTuple, Sequence, Tuple, Any
import functools

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
import optax

Array = chex.Array
PRNGKey = chex.PRNGKey


class FlowParameters(NamedTuple):
    event_shape: Sequence[int]
    num_layers: int
    hidden_sizes: Sequence[int]
    num_bins: int


class ContextConditioner(hk.Module):
    def __init__(
        self,
        event_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        num_bijector_params: int,
    ):
        super().__init__()
        self.event_shape = event_shape
        self.hidden_sizes = hidden_sizes
        self.num_bijector_params = num_bijector_params

    def __call__(self, inputs, context):
        x = jnp.concatenate((inputs, context), axis=-1)
        x = hk.Flatten(preserve_dims=-len(self.event_shape))(x)
        x = hk.nets.MLP(self.hidden_sizes, activate_final=True)(x)

        x = hk.Linear(
            np.prod(self.event_shape) * self.num_bijector_params,
        )(x)

        x = hk.Reshape(
            tuple(self.event_shape) + (self.num_bijector_params,), preserve_dims=-1
        )(x)

        return x


def make_conditioner(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
) -> hk.Sequential:
    """Creates an MLP conditioner for each layer of the flow."""

    return hk.Sequential(
        [
            hk.Flatten(preserve_dims=-len(event_shape)),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            # We initialize this linear layer to zero so that the flow is initialized
            # to the identity function.
            hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                # w_init=jnp.zeros,
                # b_init=jnp.zeros,
            ),
            hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
        ]
    )


def make_bijection(
    event_shape: Sequence[int],
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
) -> distrax.Bijector:
    """Creates the flow model."""

    # Alternating binary mask.
    mask = jnp.arange(0, np.prod(event_shape)) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

    # Number of parameters for the rational-quadratic spline:
    # - `num_bins` bin widths
    # - `num_bins` bin heights
    # - `num_bins + 1` knot slopes
    # for a total of `3 * num_bins + 1` parameters.
    num_bijector_params = 3 * num_bins + 1

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(
                event_shape, hidden_sizes, num_bijector_params
            ),
        )
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    return distrax.Chain(layers)


def make_forward(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def forward(data: Array) -> Array:
        bij = make_bijection(
            event_shape=data.shape[1:],
            num_layers=hparams.num_layers,
            hidden_sizes=hparams.hidden_sizes,
            num_bins=hparams.num_bins,
        )
        return bij.forward(data)

    return forward


def make_forward_and_log_det(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def forward_and_log_det(data: Array) -> Tuple[Array, Array]:
        bij = make_bijection(
            event_shape=data.shape[1:],
            num_layers=hparams.num_layers,
            hidden_sizes=hparams.hidden_sizes,
            num_bins=hparams.num_bins,
        )
        return bij.forward_and_log_det(data)

    return forward_and_log_det


def make_inverse(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def inverse(data: Array) -> Array:
        bij = make_bijection(
            event_shape=data.shape[1:],
            num_layers=hparams.num_layers,
            hidden_sizes=hparams.hidden_sizes,
            num_bins=hparams.num_bins,
        )
        return bij.inverse(data)

    return inverse


def make_inverse_and_log_det(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def inverse_and_log_det(data: Array) -> Tuple[Array, Array]:
        bij = make_bijection(
            event_shape=data.shape[1:],
            num_layers=hparams.num_layers,
            hidden_sizes=hparams.hidden_sizes,
            num_bins=hparams.num_bins,
        )
        return bij.inverse_and_log_det(data)

    return inverse_and_log_det


def make_log_prob_uniform(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def log_prob(data: Array) -> Array:
        event_shape = data.shape[1:]
        bij = make_bijection(
            event_shape=event_shape,
            num_layers=hparams.num_layers,
            hidden_sizes=hparams.hidden_sizes,
            num_bins=hparams.num_bins,
        )
        base = distrax.Independent(
            distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
            reinterpreted_batch_ndims=len(event_shape),
        )
        dist = distrax.Transformed(base, bij)

        return dist.log_prob(data)

    return log_prob


def make_log_prob(hparams: FlowParameters, base_distribution: distrax.Distribution):
    @hk.without_apply_rng
    @hk.transform
    def log_prob(data: Array) -> Array:
        bij = make_bijection(
            event_shape=data.shape[1:],
            num_layers=hparams.num_layers,
            hidden_sizes=hparams.hidden_sizes,
            num_bins=hparams.num_bins,
        )
        dist = distrax.Transformed(base_distribution, bij)

        return dist.log_prob(data)

    return log_prob


class Flow(struct.PyTreeNode):

    base_distribution: distrax.Distribution = struct.field(pytree_node=False)
    _log_prob: Callable = struct.field(pytree_node=False)
    _forward: Callable = struct.field(pytree_node=False)
    _inverse: Callable = struct.field(pytree_node=False)
    _forward_and_log_det: Callable = struct.field(pytree_node=False)
    _inverse_and_log_det: Callable = struct.field(pytree_node=False)
    _init: Callable = struct.field(pytree_node=False)

    event_shape: tuple[int, ...] = struct.field(pytree_node=False)

    def sample(
        self, *, params: hk.Params, seed: PRNGKey, sample_shape: Tuple[int, ...] = (1,)
    ):
        """Sample from the latent distribution.

        Parameters
        ----------
        params:
            Model parameters.
        seed: PRNGKey
            Seed for the random number generator.
        sample_shape: tuple
            Shape of the sample.

        Returns
        -------
        sample: jnp.ndarray
            Samples from latent distribution.
        """
        x = self.base_distribution.sample(seed=seed, sample_shape=sample_shape)
        return self.forward(params, x)

    def log_prob(self, params: hk.Params, z: Array):
        """Compute the log(probability) of a sample in the latent distribution.

        Parameters
        ----------
        x: jnp.ndarray
            Event in the latent distribution.

        Returns
        -------
        log_prob: jnp.ndarray
            Log(probability) of z.
        """
        return self._log_prob(params, z)

    def forward(self, params: hk.Params, x: Array):
        """Compute the push through of x through flow.

        Parameters
        ----------
        x: jnp.ndarray
            Event in the base distribution.

        Returns
        -------
        pt: jnp.ndarray
            Push-through of x.
        """
        return self._forward(params, x)

    def inverse(self, params: hk.Params, z: Array):
        """Compute the pull-back of z through flow.

        Parameters
        ----------
        x: jnp.ndarray
            Event in the latent distribution.

        Returns
        -------
        pb: jnp.ndarray
            Pull-back of z.
        """
        return self._inverse(params, z)

    def forward_and_log_det(self, params: hk.Params, x: Array):
        """Compute the push-through of x through flow and the log(determinant).

        Parameters
        ----------
        x: jnp.ndarray
            Event in the base distribution.

        Returns
        -------
        pb: jnp.ndarray
            Push-through of x.
        log_det: jnp.ndarray
            Log(determinant) of push-through.
        """
        return self._forward_and_log_det(params, x)

    def inverse_and_log_det(self, params: hk.Params, z: Array):
        """Compute the pull-back of z through flow and the log(determinant).

        Parameters
        ----------
        x: jnp.ndarray
            Event in the latent distribution.

        Returns
        -------
        pb: jnp.ndarray
            Pull-back of z.
        log_det: jnp.ndarray
            Log(determinant) of pull-back.
        """
        return self._inverse_and_log_det(params, z)

    def init(self, key):
        return self._init(key)

    @classmethod
    def create(cls, *, base: distrax.Distribution, hparams: FlowParameters):
        log_prob_ = make_log_prob(hparams, base)
        forward_ = make_forward(hparams)
        inverse_ = make_inverse(hparams)
        forward_and_log_det_ = make_forward_and_log_det(hparams)
        inverse_and_log_det_ = make_inverse_and_log_det(hparams)

        def init(key: PRNGKey) -> hk.Params:
            params = log_prob_.init(key, jnp.zeros((1, *hparams.event_shape)))
            return params

        @jax.jit
        def log_prob(params: hk.Params, x: Array) -> Array:
            return log_prob_.apply(params, x)

        @jax.jit
        def forward(params: hk.Params, x: Array) -> Array:
            return forward_.apply(params, x)

        @jax.jit
        def inverse(params: hk.Params, x: Array) -> Array:
            return inverse_.apply(params, x)

        @jax.jit
        def forward_and_log_det(params: hk.Params, x: Array) -> Array:
            return forward_and_log_det_.apply(params, x)

        @jax.jit
        def inverse_and_log_det(params: hk.Params, x: Array) -> Array:
            return inverse_and_log_det_.apply(params, x)

        return cls(
            base_distribution=base,
            _log_prob=log_prob,
            _forward=forward,
            _inverse=inverse,
            _forward_and_log_det=forward_and_log_det,
            _inverse_and_log_det=inverse_and_log_det,
            _init=init,
            event_shape=hparams.event_shape,
        )


class FlowTrainState(struct.PyTreeNode):
    params: hk.Params
    flow: Flow = struct.field(pytree_node=False)
    step: int
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def apply_gradients(self, *, grads, **kwargs):
        updates, opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)

        return self.replace(
            params=params,
            step=self.step + 1,
            opt_state=opt_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls, *, params: hk.Params, flow: Flow, optimizer: optax.GradientTransformation
    ):
        opt_state = optimizer.init(params)
        return cls(
            params=params, flow=flow, step=1, optimizer=optimizer, opt_state=opt_state
        )


@functools.partial(jax.jit, static_argnums=1)
def negative_log_loss_fn(params: hk.Params, apply_fn: Callable, x: Array) -> Array:
    loss = -jnp.mean(apply_fn(params, x))
    return loss


@functools.partial(jax.jit, static_argnums=1)
def update_flow(
    flow: Flow, tx: optax.GradientTransformation, opt_state: optax.OptState, x: Array
):
    loss, grads = jax.value_and_grad(loss_fn)(flow.params, flow.log_prob, x)
    updates, new_opt_state = tx.update(grads, opt_state)
    new_params = optax.apply_updates(flow.params, updates)
    return flow.update_params(params=new_params), new_opt_state, loss


def make_context_bijection(hparams: FlowParameters):
    """Creates the flow model."""

    event_shape = hparams.event_shape
    hidden_sizes = hparams.hidden_sizes
    num_bins = hparams.num_bins
    num_layers = hparams.num_layers

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

    def flow(context):
        # Alternating binary mask.
        mask = jnp.arange(0, np.prod(event_shape)) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters.
        num_bijector_params = 3 * num_bins + 1

        layers = []
        for _ in range(num_layers):
            conditioner = ContextConditioner(
                event_shape, hidden_sizes, num_bijector_params
            )
            layer = distrax.MaskedCoupling(
                mask=mask,
                bijector=bijector_fn,
                conditioner=lambda x: conditioner(x, context),
            )
            layers.append(layer)
            # Flip the mask after each layer.
            mask = jnp.logical_not(mask)

        return distrax.Chain(layers)

    return flow


def make_context_forward(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def forward(x, context):
        bij = make_context_bijection(hparams)
        return bij(context).forward(x)

    return forward


def make_context_inverse(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def fn(x, context):
        bij = make_context_bijection(hparams)
        return bij(context).inverse(x)

    return fn


def make_context_forward_and_log_det(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def fn(x, context):
        bij = make_context_bijection(hparams)
        return bij(context).forward_and_log_det(x)

    return fn


def make_context_inverse_and_log_det(hparams: FlowParameters):
    @hk.without_apply_rng
    @hk.transform
    def fn(x, context):
        bij = make_context_bijection(hparams)
        return bij(context).inverse_and_log_det(x)

    return fn
