"""Implementation of a standard Restricted Boltzmann Machine"""
import numpy as np
from scipy.special import expit as sigmoid
from numpy_ml.neural_nets import optimizers


def batched_array(array, batch_size):
    for first in np.arange(0, len(array), batch_size):
        yield array[first:first + batch_size]


class RestrictedBoltzmannMachine:
    """Energy based model with a bipartite interaction graph over binary visible (v) and hidden (h) nodes.
    The energy function is defined as
    E(v,h) = - a.v - b.h - v.w.h
    where a,b and w are trainable parameters.
    """

    def __init__(self, n_visible, n_hidden, seed=123, optimizer="Adam", optimzer_params=tuple()):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.w = np.ones((n_visible, n_hidden))
        self.a = np.zeros((n_visible,))
        self.b = np.zeros((n_hidden,))
        self.rng = np.random.default_rng(seed=seed)
        self.optim = self.get_optimizer(optimizer=optimizer, optimizer_params=optimzer_params)

    @staticmethod
    def get_optimizer(optimizer="Adam", optimizer_params=tuple()) -> optimizers.OptimizerBase:
        """Instantiate an optimizer from numpy_ml"""
        optim_cls = getattr(optimizers, optimizer)
        optimizer_params = dict(optimizer_params)
        return optim_cls(**optimizer_params)

    def get_rng(self, seed=None):
        """Either use the rng setup at instantiation or obtain a new one with a fixed seed"""
        if seed is None:
            return np.random.default_rng(seed)
        else:
            return self.rng

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

    def mean_negative_log_likelihood(self, v: np.ndarray):
        """Compute the mean negative log likelihood of a batch of visible states under the model.
        The log likelihood of a data point is the conditional free energy"""
        return np.mean(self.conditional_free_energy(v))

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

        return np.sum(np.outer(v, h) - np.outer(v_prime, h), axis=0)

    def compute_gradient_a(self, v, seed=None):
        """Compute the gradient of the visible linear term using contrastive divergence"""
        h = self.sample_h(v, seed)
        v_prime = self.sample_v(h, seed=seed)

        return np.sum(v - v_prime, axis=0)

    def compute_gradient_b(self, v, seed=None):
        """Compute the gradient of the hidden linear term using contrastive divergence"""
        prob_h_data = self.prob_h(v)

        h = self.sample_h(v, seed)
        v_prime = self.sample_v(h, seed)
        prob_h_model = self.prob_h(v_prime)

        return np.sum(prob_h_data - prob_h_model, axis=0)

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
        v = self.get_rng(seed).shuffle(v, axis=0)
        for v_batch in batched_array(v, batch_size=batch_size):
            self.gradient_step(v_batch)

    def train(self, v, batch_size, n_epochs):
        for e in range(n_epochs):
            self.train_epoch(v, batch_size)
