import torch
from kernels.curlfree_imq import CurlFreeIMQ
from .base import Base
import math
class NuMethod(Base):
    def __init__(self,
                 lam=None,
                 iternum=None,
                 kernel=CurlFreeIMQ(),
                 nu=1.0,
                 dtype=torch.float32):
        if lam is not None and iternum is not None:
            raise RuntimeError('Cannot specify `lam` and `iternum` simultaneously.')
        if lam is None and iternum is None:
            raise RuntimeError('Both `lam` and `iternum` are `None`.')
        if iternum is not None:
            lam = 1.0 / (iternum ** 2).to(dtype)
        else:
            iternum = int(1.0 / math.sqrt(lam)) + 1
        super().__init__(lam, kernel, dtype)
        self._nu = nu
        self._iternum = iternum

    def fit(self, samples, kernel_hyperparams=None):
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = samples.shape[-2]
        d = samples.shape[-1]
        #K_div
        K_op, K_div = self._kernel.kernel_operator(samples, samples,
                kernel_hyperparams=kernel_hyperparams)

        H_dh = torch.mean(K_div, axis=-2).reshape([M * d, 1])

        def get_next(t, a, pa, c, pc):
            ft = float(t)
            nu = self._nu
            u = (ft - 1.) * (2. * ft - 3.) * (2. * ft + 2. * nu - 1.) /((ft + 2. * nu - 1.) * (2. * ft + 4. * nu - 1.) * (2. * ft + 2. * nu - 3.))
            w = 4. * (2. * ft + 2. * nu - 1.) * (ft + nu - 1.) / ((ft + 2. * nu - 1.) * (2. * ft + 4. * nu - 1.))
            nc = (1. + u) * c - w * (a * H_dh + K_op['apply'](c)) / M - u * pc
            na = (1. + u) * a - u * pa - w
            return (t + 1, na, a, nc, c)

        a1 = -(4. * self._nu + 2) / (4. * self._nu + 1)
        ret = get_next(2, a1, 0., torch.zeros_like(H_dh), torch.zeros_like(H_dh))

        while ret[0] <= self._iternum:
            ret = get_next(*ret)

        self._coeff = (ret[1], ret[3])

    def _compute_energy(self, x):
        Kxq, div_xq = self._kernel.kernel_energy(x, self._samples,
                kernel_hyperparams=self._kernel_hyperparams)
        Kxq = Kxq.reshape([x.shape[-2], -1])
        div_xq = torch.mean(div_xq, axis=-1) * self._coeff[0]
        energy = torch.matmul(Kxq, self._coeff[1]).reshape([-1]) + div_xq
        return energy
    
    def compute_gradients(self, x):
        d = x.shape[-1]
        Kxq_op, div_xq = self._kernel.kernel_operator(x, self._samples,
                kernel_hyperparams=self._kernel_hyperparams)
        div_xq = torch.mean(div_xq, dim=-2) * self._coeff[0]
        grads = Kxq_op['apply'](self._coeff[1])
        grads = grads.reshape([-1, d]) + div_xq
        return grads