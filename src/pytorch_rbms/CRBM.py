# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 08:57:36 2023

@author: COND04848
"""

import torch
from tqdm.autonotebook import tqdm
from pytorch_rbms.utils import batched_array, batched_outer


class ContinuousRestrictedBoltzmannMachine:
    def __init__(
        self,
        n_visible,
        n_hidden,
        v_sampling="inverse",
        lr=1.0e-3,
        seed=None,
        n_CD_steps=1,
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.w = torch.nn.Parameter(
            torch.zeros((n_visible, n_hidden), dtype=torch.float), requires_grad=False
        )
        self.a = torch.nn.Parameter(
            torch.zeros((n_visible,), dtype=torch.float), requires_grad=False
        )
        self.b = torch.nn.Parameter(
            torch.zeros((n_hidden,), dtype=torch.float), requires_grad=False
        )
        self.optim = torch.optim.Adam([self.w, self.a, self.b], lr=lr)
        self.rng = self.get_rng(seed=seed)
        self.v_sampling = v_sampling
        self.k = n_CD_steps

    def reset_optim(self, lr=1.0e-3):
        self.optim = torch.optim.Adam([self.w, self.a, self.b], lr=lr)

    def set_lr(self, lr):
        for g in self.optim.param_groups:
            g["lr"] = lr

    def scale_lr(self, scale=1.0):
        for g in self.optim.param_groups:
            g["lr"] *= scale

    def get_rng(self, seed=None):
        if seed is None:
            return torch.default_generator
        else:
            return torch.manual_seed(seed)

    def initialize_w(self, scale=1.0, seed=None):
        self.w.data = torch.normal(
            mean=torch.zeros((self.n_visible, self.n_hidden)), std=0.1 * scale
        )

    def initialize_a(self, v=None, seed=None):
        self.a.data = torch.zeros((self.n_visible,))
        if v is not None:
            pv = torch.mean(v, 0)
            self.a.data = (pv / (1 - pv)).log()

    def initialize_b(self, scale=1.0, seed=None):
        self.b.data = torch.zeros((self.n_hidden,))

    def initialization(self, scale=1.0, v=None, seed=None):
        self.initialize_w(scale=scale, seed=seed)
        self.initialize_a(v=v, seed=seed)
        self.initialize_b(scale=scale, seed=seed)

    def xavier_initialize_w(self, seed=None):
        self.w = torch.normal(
            mean=torch.zeros((self.n_visible, self.n_hidden)),
            std=4 / torch.sqrt(torch.tensor([self.n_visible * self.n_hidden])),
            generator=self.rng,
        )

    def xavier_initialize_a(self, seed=None):
        self.a = torch.normal(
            mean=torch.zeros((self.n_visible,)),
            std=4 / torch.sqrt(torch.tensor([self.n_visible])),
            generator=self.rng,
        )

    def xavier_initialize_b(self, seed=None):
        self.b = torch.normal(
            mean=torch.zeros((self.n_hidden,)),
            std=4 / torch.sqrt(torch.tensor([self.n_hidden])),
            generator=self.rng,
        )

    def xavier_initialization(self, seed=None):
        self.xavier_initialize_w(seed)
        self.xavier_initialize_a(seed)
        self.xavier_initialize_b(seed)

    def prob_h(self, v):
        assert v.size(-1) == self.w.size(0)
        return torch.sigmoid(self.b + torch.tensordot(v, self.w, dims=([-1], [0])))

    def sample_h(self, v, seed=None):
        return torch.bernoulli(self.prob_h(v), generator=self.rng)

    def sample_v(self, h, seed=None):
        if self.v_sampling.lower() == "normal" or self.v_sampling.lower() == "gaussian":
            ten = self.a + torch.tensordot(h, self.w, dims=([-1], [1]))
            return torch.normal(ten, torch.ones(ten.size()))
        elif self.v_sampling.lower() == "inverse" or self.v_sampling.lower() is None:
            size = (h.size(dim=0), self.n_visible)  # Batch size, n_visible
            U = torch.rand(size=size, generator=self.rng, dtype=torch.float)
            alpha = -self.a - torch.tensordot(h, self.w, dims=([-1], [1]))

            return -torch.log(1 - U * (1 - torch.exp(-alpha))) / alpha
        else:
            Exception(
                """Please specify sampling for visible nodes,
                can be 'gaussian' or `inverse`"""
            )

    def sample_initial(self, batch_size=10, v_start=None, seed=None):
        if v_start is None:
            v_start = torch.rand(size=(batch_size, self.n_visible), generator=self.rng)
        else:
            batch_size = v_start.size(0)
        return v_start, batch_size

    def step_gibbs(self, v, mask_updates=None, seed=None):
        h = self.sample_h(v, seed)
        v_new = self.sample_v(h, seed)
        if mask_updates is not None:
            assert mask_updates.size() == v.size(-1)
            v_new = mask_updates * v + (1 - mask_updates) * v_new
        return v_new, h

    def sample_gibbs(
        self,
        batch_size=10,
        n_steps=100,
        n_burn=10,
        return_hidden=False,
        v_start=None,
        mask_updates=None,
        seed=None,
    ):
        v, batch_size = self.sample_initial(
            batch_size=batch_size, v_start=v_start, seed=seed
        )
        sample_v = torch.zeros((n_steps * batch_size, self.n_visible))
        if return_hidden:
            sample_h = torch.zeros((n_steps * batch_size, self.n_hidden))
        for i in range(n_burn):
            v, h = self.step_gibbs(v, mask_updates=mask_updates, seed=seed)
        for i in range(n_steps):
            v, h = self.step_gibbs(v, mask_updates=mask_updates, seed=seed)
            sample_v[i * batch_size : (i + 1) * batch_size] = torch.clone(v)
            if return_hidden:
                sample_h[i * batch_size : (i + 1) * batch_size] = torch.clone(h)
        if return_hidden:
            return sample_v, sample_h
        else:
            return sample_v

    def compute_gradient_a(self, v, seed=None):
        v_k = v
        for i in range(self.k):
            v_k, h_k = self.step_gibbs(v_k)

        return torch.sum(v - v_k, dim=0)

    def compute_gradient_b(self, v, seed=None):
        v_k = v
        for i in range(self.k):
            v_k, h_k = self.step_gibbs(v_k)

        prob_h_data = self.prob_h(v)
        prob_h_model = self.prob_h(v_k)

        return torch.sum(prob_h_data - prob_h_model, dim=0)

    def compute_gradient_w(self, v, seed=None):
        v_k = v
        for i in range(self.k):
            v_k, h_k = self.step_gibbs(v_k)

        prob_h_data = self.prob_h(v)
        prob_h_model = self.prob_h(v_k)

        return torch.sum(
            batched_outer(v, prob_h_data) - batched_outer(v_k, prob_h_model), dim=0
        )

    def SGD_step(self, v, lr=0.001, weight_decay=0, seed=None):
        gradient_w = self.compute_gradient_w(v, seed)
        gradient_a = self.compute_gradient_a(v, seed)
        gradient_b = self.compute_gradient_b(v, seed)

        self.optim.zero_grad()
        self.a.grad = -1 * gradient_a  # Adam optimizer assumes you want to
        self.b.grad = -1 * gradient_b  # minimise the loss. However we want to maximise
        self.w.grad = -1 * gradient_w  # likelihood, so we negate the loss.
        self.optim.step()

    def eval_cov_l2(self, test_set, predicted_set):
        cov_test = torch.corrcoef(torch.transpose(test_set, 0, -1)) - torch.eye(
            test_set.size(-1)
        )
        cov_pred = torch.corrcoef(torch.transpose(predicted_set, 0, -1)) - torch.eye(
            predicted_set.size(-1)
        )

        return torch.linalg.norm(cov_pred - cov_test) / torch.linalg.norm(cov_test)

    def eval_cov_l2_gibbs(
        self, test_set, batch_size=10, n_steps=1000, n_burn=1000, seed=None
    ):
        predicted_set = self.sample_gibbs(
            batch_size=batch_size,
            n_steps=n_steps,
            n_burn=n_burn,
            return_hidden=False,
            seed=seed,
        )

        return self.eval_cov_l2(test_set=test_set, predicted_set=predicted_set).item()

    def train_epoch(self, v, batch_size, seed=None, lr=0.001, weight_decay=0):
        u = v[torch.randperm(v.size(0))]
        for v_batch in batched_array(u, batch_size=batch_size):
            self.SGD_step(v_batch, lr=lr, weight_decay=weight_decay, seed=seed)

    def train(self, v, batch_size, n_epochs, lr=0.001, weight_decay=0, seed=None):
        covl2_history = []
        covl2_history.append(self.eval_cov_l2_gibbs(test_set=v))
        progress_bar = tqdm(
            range(n_epochs), desc=f"Cov L2 Diff: {covl2_history[-1]:.2e}"
        )
        for e in progress_bar:
            self.train_epoch(
                v, batch_size=batch_size, seed=seed, lr=lr, weight_decay=weight_decay
            )
            covl2_history.append(self.eval_cov_l2_gibbs(test_set=v))
            progress_bar.set_description(
                desc=f"Cov L2 Diff: {covl2_history[-1]:.2e}", refresh=True
            )
        return covl2_history

    @property
    def parameters(self):
        return [self.a, self.b, self.w]
