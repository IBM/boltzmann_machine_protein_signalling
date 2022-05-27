from bops.rbm import RestrictedBoltzmannMachine


class ConstrainedRestrictedBoltzmannMachine(RestrictedBoltzmannMachine):
    def __init__(
            self,
            n_visible,
            n_hidden,
            adjacency_matrix,
            seed=None,
            optimizer="Adam",
            optimizer_params=tuple(),
    ):
        super(ConstrainedRestrictedBoltzmannMachine, self).__init__(
            n_visible=n_visible,
            n_hidden=n_hidden,
            seed=seed,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
        )
        assert adjacency_matrix.shape == self.w.shape, f"The adjacency matrix should have shape ({n_visible}, {n_hidden})"
        assert set(adjacency_matrix.flatten).issubset({0., 1.}), "The adjacency matrix should contain only 1. and 0."
        self.adjacency_matrix = adjacency_matrix

    def xavier_initialize_w(self, seed=None):
        super(ConstrainedRestrictedBoltzmannMachine, self).xavier_initialize_w(seed=seed)
        self.w = self.w * self.adjacency_matrix

    def compute_gradient_w(self, v, seed=None):
        gradient = super(ConstrainedRestrictedBoltzmannMachine, self).compute_gradient_w(v=v, seed=seed)
        return gradient * self.adjacency_matrix
