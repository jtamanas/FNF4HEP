"""
Module for constructing a FairFlow.
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

from fax._src.flow import Flow, FlowParameters, FlowTrainState

Array = chex.Array
PRNGKey = Array


class FairFlows(struct.PyTreeNode):
    flows: Tuple[Flow, ...]

    def update_params(self, *, params: Tuple):
        flows = jax.tree_util.tree_map(
            lambda f, p: f.update_params(params=p), zip(self.flows, params)
        )
        return self.replace(flows=flows)

    def log_prob(self, x: Array, i: int):
        """Compute the log(probability) of a forward pass.

        Parameters
        ----------
        x: jnp.ndarray
            Event in the base distribution.

        Returns
        -------
        log_prob: jnp.ndarray
            Log(probability) of push-through of x.
        """
        return self.flows[i].log_prob(x)

    def forward(self, x: Array, i: int):
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
        return self.flows[i].forward(x)

    def inverse(self, x: Array, i: int):
        """Compute the pull-back of x through flow.

        Parameters
        ----------
        x: jnp.ndarray
            Event in the target distribution.

        Returns
        -------
        pb: jnp.ndarray
            Pull-back of x.
        """
        return self.flows[i].inverse(x)

    def forward_and_log_det(self, x: Array, i: int):
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
        return self.flows[i].forward_and_log_det(x)

    def inverse_and_log_det(self, x: Array, i: int):
        """Compute the pull-back of x through flow and the log(determinant).

        Parameters
        ----------
        x: jnp.ndarray
            Event in the target distribution.

        Returns
        -------
        pb: jnp.ndarray
            Pull-back of x.
        log_det: jnp.ndarray
            Log(determinant) of pull-back.
        """
        return self.flows[i].inverse_and_log_det(x)

    @classmethod
    def create(
        cls,
        *,
        base: distrax.Distribution,
        key: Array,
        event_shape: Sequence[int],
        hparams: Tuple[FlowParameters, ...]
    ):
        flows = []
        for hp in hparams:
            key, rng = jax.random.split(key, 2)
            flows.append(
                Flow.create(base=base, key=rng, event_shape=event_shape, hparams=hp)
            )

        return cls(flows=flows)


class FairFlowsTrainState(struct.PyTreeNode):
    flows: Tuple[FlowTrainState, ...]
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState

    def update_params(self, *, params: Tuple):
        flows = jax.tree_util.tree_map(
            lambda f, p: f.update_params(p), zip(self.flows, params)
        )
        return self.replace(flows=flows)

    def log_prob(self, x: Array, i: int):
        return self.flows[i].flow.log_prob(x)

    def forward(self, x: Array, i: int):
        return self.flows[i].flow.forward(x)

    def inverse(self, x: Array, i: int):
        return self.flows[i].flow.inverse(x)

    def forward_and_log_det(self, x: Array, i: int):
        return self.flows[i].flow.forward_and_log_det(x)

    def inverse_and_log_det(self, x: Array, i: int):
        return self.flows[i].flow.inverse_and_log_det(x)

    @classmethod
    def create(
        cls,
        *,
        base: distrax.Distribution,
        key: Array,
        event_shape: Sequence[int],
        hparams: Tuple[FlowParameters, ...]
    ):
        flows = FairFlows.create(
            base=base, key=key, event_shape=event_shape, hparams=hparams
        )

        return cls(flows=flows)


def fair_flows_loss_fn(
    params: Tuple[hk.Params, ...],
    apply_fns: Tuple[Callable, ...],
    xs: Tuple[Array, ...],
):

    loss = 0.0
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            log_prob1 = apply_fns[i](params[i], xs[i])
            log_prob2 = apply_fns[j](params[j], xs[j])
            p1 = jnp.exp(log_prob1)
            p2 = jnp.exp(log_prob2)
            loss = loss + jnp.mean(p1 * (log_prob2 - log_prob1))
            loss = loss + jnp.mean(p2 * (log_prob1 - log_prob2))
