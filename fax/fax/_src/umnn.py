import functools

import jax
import jax.numpy as jnp
import numpy as np
import distrax


class UnconstrainedMonotonic(distrax.Bijector):
    def __init__(self, derivative, embedding, bias, n, niter):
        self._derivative = derivative
        self._embedding = embedding
        self._bias = bias
        self._n = n
        self._niter = niter
        super().__init__(1)

    def forward_and_log_det(self, x):
        z, logdet = umnn_forward_and_log_det(
            self._derivative, self._embedding, self._bias, x, self._n
        )
        return z, logdet

    def inverse_and_log_det(self, z):
        x, loginvdet = umnn_inverse_and_log_det(
            self._derivative, self._embedding, self._bias, z, self._n, self._niter
        )
        return x, loginvdet


def clenshaw_curtis_quadrature_weights(n):
    """Compute the Clenshaw-Curtis quadrature weights and abscissa.

    Parameters
    ----------
    n: int
        Number of weights and abscissa to generate.

    Returns
    -------
    abscissa: array
        Locations of the quadrature points.
    weights: array
        Weights of the quadrature points.
    """
    k = np.arange(0, n // 2 + 1)

    dmat = 2 / n * np.cos(np.outer(k, k) * np.pi / (n // 2))
    dmat[:, 0] *= 0.5
    dmat[:, -1] *= 0.5

    dvec = 2.0 / (1.0 - (2.0 * k) ** 2)
    dvec[[0, -1]] *= 0.5

    w = jnp.array(dmat.T @ dvec)
    x = jnp.array(np.cos(k * np.pi / n))

    return x, w


@functools.partial(jax.vmap, in_axes=(None, None, None, 0, None), out_axes=0)
def umnn_forward_and_log_det(f, h, b, x, n):
    """Compute forward pass through an Unconstrained Monotonic Neutral Network bijector.

    Parameters
    ----------
    f: callable
        Function to integrate. Should have signature f(t, h(x)).
    h: callable
        Embedding function to transform x. The output of h is used as an input
        to f and b.
    b: callable
        Bias function for the integral. Signature should be b(x').
    x: ndarray
        Array of random variables to transform.
    n: int
        Number of Clenshaw-Curtis quadrature points to use.
    """
    ts, wgts = clenshaw_curtis_quadrature_weights(n)

    # Mask to make autoregressive
    xa = jnp.tri(x.shape[0], x.shape[0]) * x
    # Transform abscissa from (-1, 1) to (a, b)
    y = jnp.expand_dims(x, 1)
    tp = y * 0.5 * (ts + 1.0)
    tm = y * 0.5 * (-ts + 1.0)

    hx = h(xa)
    fx = 0.5 * y * (f(tp, hx) + f(tm, hx))
    bx = b(hx)
    z = jnp.sum(wgts * fx, axis=1) + bx
    jac = 0.5 * y * (f(y, hx) + f(-y, hx))
    logdet = jnp.prod(jnp.diag(jac), axis=0)

    return z, logdet


@functools.partial(jax.vmap, in_axes=(None, None, None, 0, None, None), out_axes=0)
def umnn_inverse_and_log_det(f, h, b, z, n, niters):
    """Compute the inverse of a Unconstrained Monotonic Neutral Network bijector.

    Parameters
    ----------
    f: callable
        Function to integrate. Should have signature f(t, h(x)).
    h: callable
        Embedding function to transform x. The output of h is used as an input
        to f and b.
    b: callable
        Bias function for the integral. Signature should be b(x').
    x: ndarray
        Array of random variables to transform.
    n: int
        Number of Clenshaw-Curtis quadrature points to use.
    niter: int
        Number of Newton iterations.
    """
    ts, wgts = clenshaw_curtis_quadrature_weights(n)

    def newton(_, state, j):
        x = state
        # Transform abscissa from (-1, 1) to (a, b)
        t = jnp.expand_dims(0.5 * (ts + 1.0), (1,)) * x
        hx = h(x)
        fx = f(t, hx) + f(-t, hx)
        bx = b(hx)
        g = (wgts.T @ fx + bx) - z[j]
        dg = f(x, hx) + f(-x, hx)
        # Newton update
        x = x - dg / g

        return x

    def invert_j(j, x):
        def newton_j(i, state):
            return newton(i, state, j)

        _, x = jax.lax.fori_loop(0, niters, newton_j, (x,))
        return x

    x = jnp.zeros_like(z)
    x = jax.lax.fori_loop(0, n, invert_j, (x,))

    # Mask to make autoregressive
    xa = jnp.tri(x.shape[0], x.shape[0]) * x

    hx = h(x)
    jac = f(xa, hx) - f(-xa, hx)
    inv = jnp.linalg.inv(jac)
    logdet = jnp.prod(jnp.diag(inv))

    return x, logdet
