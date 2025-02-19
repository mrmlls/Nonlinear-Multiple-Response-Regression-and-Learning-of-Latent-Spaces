from .base import Base
from .utils import median_heuristic
import torch # type: ignore

class SquareCurlFree(Base):

    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__('curl-free', kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives(self, x, y, kernel_hyperparams):
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams
        x_m = x.unsqueeze(-2)  # [M, 1, d]
        y_m = y.unsqueeze(-3)  # [1, N, d]
        r = x_m - y_m  # [M, N, d]
        norm_rr = torch.sum(r * r, -1)  # [M, N]
        return self._gram_derivatives_impl(r, norm_rr, kernel_width)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        raise NotImplementedError('`_gram_derivatives` not implemented.')

    def kernel_energy(self, x, y, kernel_hyperparams=None, compute_divergence=True):
        d, M, N = x.shape[-1], x.shape[-2], y.shape[-2]
        r, norm_rr, G_1st, G_2nd, _ = self._gram_derivatives(x, y, kernel_hyperparams)

        energy_k = -2. * G_1st.unsqueeze(-1) * r

        if compute_divergence:
            divergence = 2 * d * G_1st + 4. * norm_rr * G_2nd
            return energy_k, divergence
        return energy_k
    
    def kernel_operator(self, x, y, kernel_hyperparams=None, compute_divergence=True):
        d, M, N = x.shape[-1], x.shape[-2], y.shape[-2]
        r, norm_rr, G_1st, G_2nd, G_3rd = self._gram_derivatives(x, y, kernel_hyperparams)
        G_1st = G_1st.unsqueeze(-1)   # [M, N, 1]
        G_2nd = G_2nd.unsqueeze(-1)   # [M, N, 1]

        if compute_divergence:
            coeff = (d + 2) * G_2nd + 2. * (norm_rr * G_3rd).unsqueeze(-1)
            divergence = 4. * coeff * r

        def kernel_op(z):
            # z: [N * d, L]
            L = z.shape[-1]
            z = z.reshape([1, N, d, L])            # [1, N, d, L]
            hat_r = r.unsqueeze(-1)                # [M, N, d, 1]
            dot_rz = torch.sum(z * hat_r, dim=-2)  # [M, N, L]
            coeff = -4. * G_2nd * dot_rz           # [M, N, L]
            ret = coeff.unsqueeze(-2) * hat_r - 2. * G_1st.unsqueeze(-1) * z
            return ret.sum(dim=-3).reshape([M * d, L])

        def kernel_adjoint_op(z):
            # z: [M * d, L]
            L = z.shape[-1]
            z = z.reshape([M, 1, d, L])            # [M, 1, d, L]
            hat_r = r.unsqueeze(-1)                # [M, N, d, 1]
            dot_rz = torch.sum(z * hat_r, dim=-2)  # [M, N, L]
            coeff = -4. * G_2nd * dot_rz           # [M, N, L]
            ret = coeff.unsqueeze(-2) * hat_r - 2. * G_1st.unsqueeze(-1) * z
            return ret.sum(dim=-4).reshape([N * d, L])

        def kernel_mat(flatten):
            Km = r.unsqueeze(-1) * r.unsqueeze(-2)
            K = -2. * G_1st.unsqueeze(-1) * torch.eye(d) - 4. * G_2nd.unsqueeze(-1) * Km
            if flatten:
                K = K.permute(0, 2, 1, 3).reshape([M * d, N * d])
            return K

        op = {
            "shape": [M * d, N * d],
            "dtype": x.dtype,
            "apply": kernel_op,
            "apply_adjoint": kernel_adjoint_op,
            "kernel_matrix": kernel_mat,
        }

        if compute_divergence:
            return op, divergence
        return op
