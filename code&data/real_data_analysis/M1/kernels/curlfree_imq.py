import torch # type: ignore
from .square_curlfree import SquareCurlFree
from .utils import median_heuristic



def random_choice(inputs, n_samples):
    uniform_prob = torch.ones(inputs.shape[0]).unsqueeze(0)
    ind = torch.multinomial(uniform_prob, n_samples).squeeze(0)
    return inputs[ind]

def conjugate_gradient(operator,
                       rhs,
                       x=None,
                       tol=1e-4,
                       max_iter=40):
    class CGState:
        def __init__(self, i, x, r, p, gamma):
            self.i = i
            self.x = x
            self.r = r
            self.p = p
            self.gamma = gamma

    def stopping_criterion(i, state):
        return i < max_iter and torch.norm(state.r) > tol

    def cg_step(i, state):
        z = operator(state.p)
        alpha = state.gamma / torch.sum(state.p * z)
        x = state.x + alpha * state.p
        r = state.r - alpha * z
        gamma = torch.sum(r * r)
        beta = gamma / state.gamma
        p = r + beta * state.p
        return i + 1, CGState(i + 1, x, r, p, gamma)

    n = operator.shape[1:]
    rhs = rhs.unsqueeze(-1)
    if x is None:
        x = torch.zeros(n).unsqueeze(-1)
        r0 = rhs
    else:
        x = x.unsqueeze(-1)
        r0 = rhs - operator(x)

    p0 = r0
    gamma0 = torch.sum(r0 * p0)
    tol *= torch.norm(r0)
    i = 0
    state = CGState(i=i, x=x, r=r0, p=p0, gamma=gamma0)
    while stopping_criterion(i, state):
        i, state = cg_step(i, state)
    return CGState(
            state.i,
            x=state.x.squeeze(),
            r=state.r.squeeze(),
            p=state.p.squeeze(),
            gamma=state.gamma)



class CurlFreeIMQ(SquareCurlFree):
    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        inv_sqr_sigma = 1.0 / torch.square(sigma)
        imq = torch.rsqrt(1.0 + norm_rr * inv_sqr_sigma) # [M, N]
        imq_2 = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        G_1st = -0.5 * imq_2 * inv_sqr_sigma * imq
        G_2nd = -1.5 * imq_2 * inv_sqr_sigma * G_1st
        G_3rd = -2.5 * imq_2 * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd
