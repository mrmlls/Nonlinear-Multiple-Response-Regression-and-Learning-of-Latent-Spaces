import torch # type: ignore

def median_heuristic(x, y):
    n = x.shape[-2]
    m = y.shape[-2]
    x_expand = x.unsqueeze(-2)
    y_expand = y.unsqueeze(-3)
    pairwise_dist = torch.sqrt(torch.sum(torch.square(x_expand - y_expand), dim=-1))
    k = n * m // 2
    top_k_values = torch.topk(pairwise_dist.view(-1, n * m), k=k).values
    kernel_width = top_k_values[:, -1].view(x.shape[:-2])
    return kernel_width.detach()