"""Deep ensembles (Lakshminarayanan et al., 2017) for prediction intervals (Weeks 5-6).

Train `ensemble_size` networks from different random initializations. Each network
outputs a mean and a variance and is trained with the Gaussian negative log-likelihood
(torch.nn.GaussianNLLLoss). Combine the members as a uniform Gaussian mixture:
    mu  = mean_m(mu_m)
    var = mean_m(var_m + mu_m**2) - mu**2
"""


class DeepEnsemble:
    def __init__(self, make_member, ensemble_size, seed):
        self.make_member = make_member  # callable returning a fresh nn.Module
        self.ensemble_size = ensemble_size
        self.seed = seed
        self.members = []

    def fit(self, train_inputs, y_log, cfg):
        raise NotImplementedError

    def predict_interval(self, inputs, level):
        """Return (mean, lower, upper) in cycles."""
        raise NotImplementedError
