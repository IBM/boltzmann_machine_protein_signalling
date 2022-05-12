"""Implementation of a standard Restricted Boltzmann Machine"""
import numpy as np
from scipy.special import expit as sigmoid
from numpy_ml.neural_nets import optimizers
from tqdm.autonotebook import tqdm


def batched_array(array, batch_size):
    for first in np.arange(0, len(array), batch_size):
        yield array[first:first + batch_size]


def batched_outer(a, b):
    """Compute the outer product of two batches of vectors
    i.e. a_ni, b_nj -> a_ni * b_nj

    Parameters
    ----------
    a: np.ndarray
        shape NxA
    b: np.ndarray
        shape NxB

    Returns
    -------
    np.ndarray
        shape NxAxB
    """
    return np.einsum("ni,nj->nij", a, b)


class RestrictedBoltzmannMachine:
    """Energy based model with a bipartite interaction graph over binary visible (v) and hidden (h) nodes.
    The energy function is defined as
    E(v,h) = - a.v - b.h - v.w.h
    where a,b and w are trainable parameters.
    """

    def __init__(self, n_visible, n_hidden, seed=123, optimizer="Adam", optimzer_params=tuple()):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.w = np.zeros((n_visible, n_hidden))
        self.a = np.zeros((n_visible,))
        self.b = np.zeros((n_hidden,))
        self.rng = np.random.default_rng(seed=seed)
        self.optim = self.get_optimizer(optimizer=optimizer, optimizer_params=optimzer_params)

    def xavier_initialize_w(self, seed=None):
        self.w = self.get_rng(seed).normal(loc=0, scale=4 / np.sqrt(self.n_hidden * self.n_visible), size=self.w.shape)

    def xavier_initialize_a(self, seed=None):
        self.a = self.get_rng(seed).normal(loc=0, scale=4 / np.sqrt(self.n_visible), size=self.a.shape)

    def xavier_initialize_b(self, seed=None):
        self.b = self.get_rng(seed).normal(loc=0, scale=4 / np.sqrt(self.n_hidden), size=self.b.shape)

    def xavier_initialization(self, seed=None):
        self.xavier_initialize_w(seed)
        self.xavier_initialize_a(seed)
        self.xavier_initialize_b(seed)

    @staticmethod
    def get_optimizer(optimizer="Adam", optimizer_params=tuple()) -> optimizers.OptimizerBase:
        """Instantiate an optimizer from numpy_ml"""
        optim_cls = getattr(optimizers, optimizer)
        optimizer_params = dict(optimizer_params)
        return optim_cls(**optimizer_params)

    def get_rng(self, seed=None):
        """Either use the rng setup at instantiation or obtain a new one with a fixed seed"""
        if seed is None:
            return self.rng
        else:
            return np.random.default_rng(seed)

    def reset_rng(self, seed):
        """Reset the default rng with a new seed"""
        self.rng = self.get_rng(seed)

    def prob_h(self, v: np.ndarray):
        """Compute the probability of all hidden nodes being equal to 1 given all the visible nodes"""
        return sigmoid(
            np.tensordot(v, self.w, axes=[[-1], [0]])  # allows both batched an non-batched entries.
            + self.b)

    def prob_v(self, h: np.ndarray):
        """Compute the probability of all visible nodes being equal to 1 given all the hidden nodes"""
        return sigmoid(
            np.tensordot(h, self.w, axes=[[-1], [1]])  # allows both batched an non-batched entries.
            + self.a)

    def conditional_free_energy(self, v: np.ndarray):
        """Compute the free energy of a batch of states with fixed visible nodes"""
        # See Hinton's A Practical Guide to Training Restricted Boltzmann Machines
        # version 1, page 17, equation 25
        # F = - ai vi - sum_j log(1+exp(hmean_j)) where hmean_j = vi wij + bj
        return - np.tensordot(v, self.a, [[-1], [0]]) \
               - np.sum(np.log(1 + np.exp(np.tensordot(v, self.w, axes=[[-1], [0]]) + self.b)), axis=-1)

    def boltzmann_factor(self, v: np.ndarray):
        """Compute the Boltzmann factor (un-normalized probability) for a batch of visible states.
        The Boltzmann factor is exp(-F(v)) where F is the free energy of the visible state v
        """
        return np.exp(-self.conditional_free_energy(v))

    def sample_initial_batch(self, batch_size=10, v_start=None, seed=None):
        """Provide an appropriate batch of vectors"""
        if v_start is None:
            v_start = self.get_rng(seed=seed).integers(size=(batch_size, self.n_visible), low=0, high=1, endpoint=True)
        else:
            batch_size = v_start.shape[0]

        return v_start, batch_size

    def sample_gibbs(self, batch_size=10, n_steps=100, n_burn=10, return_hidden=False, v_start=None, seed=None):
        """Sample using the Gibbs algorithm: start from a random configuration of visible variables and alternatively
        sample h and v from each other.

        All operations are performed batch-wise and the total number of samples will be
        (batch_size * n_steps) to allow for vectorization
        """
        v, batch_size = self.sample_initial_batch(batch_size=batch_size, v_start=v_start, seed=seed)

        # Creating arrays for the results. They are filled with nans to easily diagnose issues
        sample_v = np.empty((n_steps * batch_size, self.n_visible))
        sample_v.fill(np.nan)
        if return_hidden:
            sample_h = np.empty((n_steps * batch_size, self.n_hidden))
            sample_h.fill(np.nan)
        for i in range(n_burn):
            h = self.sample_h(v, seed)
            v = self.sample_v(h, seed)

        for i in range(n_steps):
            h = self.sample_h(v, seed)
            v = self.sample_v(h, seed)
            sample_v[i * batch_size:(i + 1) * batch_size] = v.copy()
            if return_hidden:
                sample_h[i * batch_size:(i + 1) * batch_size] = h.copy()

        if return_hidden:
            return sample_v, sample_h
        else:
            return sample_v

    def metropolis_step(self, v, seed=None):
        v_new = self.get_rng(seed).integers(0, 1, size=v.shape, endpoint=True)
        # Get un-normalized probabilities
        p_v = self.boltzmann_factor(v)
        p_v_new = self.boltzmann_factor(v_new)

        acceptance_probability = p_v_new / p_v
        accept = self.sample_binary(acceptance_probability, seed=seed)
        accept = accept[:, np.newaxis]
        return accept * v_new + (1 - accept) * v

    def sample_metropolis(self, batch_size=10, n_steps=100, n_burn=10, v_start=None, seed=None):
        """Sample using the Metropolis-Hastings algorithm. This does not require sampling the hidden variables
        as we can compute the probability of a visible state using the free energy formula"""
        v, batch_size = self.sample_initial_batch(batch_size=batch_size, v_start=v_start, seed=seed)
        sample_v = np.empty((n_steps * batch_size, self.n_visible))
        sample_v.fill(np.nan)
        for i in range(n_burn):
            v = self.metropolis_step(v, seed)

        for i in range(n_steps):
            v = self.metropolis_step(v, seed)
            sample_v[i * batch_size:(i + 1) * batch_size] = v.copy()

        return sample_v

    def sample_binary(self, probas: np.ndarray, seed: int = None):
        """Sample independent binary RV from an array of probability for each to be 1"""
        draws = self.get_rng(seed=seed).uniform(0, 1, probas.shape)
        return (draws <= probas).astype(float)

    def sample_h(self, v, seed=None):
        """Sample the hidden variables given the visible ones"""
        probas = self.prob_h(v)
        return self.sample_binary(probas=probas, seed=seed)

    def sample_v(self, h, seed=None):
        """Sample the visible variables given the hidden ones"""
        probas = self.prob_v(h)
        return self.sample_binary(probas=probas, seed=seed)

    def compute_gradient_w(self, v, seed=None):
        """Compute the gradient of the coupling term using contrastive divergence"""
        h = self.sample_h(v, seed)
        v_prime = self.sample_v(h, seed)

        return np.sum(batched_outer(v_prime, h) - batched_outer(v, h), axis=0)

    def compute_gradient_a(self, v, seed=None):
        """Compute the gradient of the visible linear term using contrastive divergence"""
        h = self.sample_h(v, seed)
        v_prime = self.sample_v(h, seed=seed)

        return np.sum(v_prime - v, axis=0)

    def compute_gradient_b(self, v, seed=None):
        """Compute the gradient of the hidden linear term using contrastive divergence"""
        prob_h_data = self.prob_h(v)

        h = self.sample_h(v, seed)
        v_prime = self.sample_v(h, seed)
        prob_h_model = self.prob_h(v_prime)

        return np.sum(prob_h_model - prob_h_data, axis=0)

    def gradient_step(self, v, seed=None):
        """Update the parameters using a minibatch of visible variables"""
        gradient_w = self.compute_gradient_w(v, seed)
        gradient_a = self.compute_gradient_a(v, seed)
        gradient_b = self.compute_gradient_b(v, seed)

        self.w = self.optim.update(self.w, gradient_w, "w")
        self.a = self.optim.update(self.a, gradient_a, "a")
        self.b = self.optim.update(self.b, gradient_b, "b")

    def train_epoch(self, v, batch_size, seed=None):
        """Train by going once over a batch of visible variables, dividing it into minibatches"""
        self.get_rng(seed).shuffle(v, axis=0)
        for v_batch in batched_array(v, batch_size=batch_size):
            self.gradient_step(v_batch)

    def train(self, v, batch_size, n_epochs):
        for e in tqdm(range(n_epochs)):
            self.train_epoch(v, batch_size)
