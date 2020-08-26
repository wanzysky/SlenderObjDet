import torch

def smooth_l1_loss_with_weight(
    input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, beta: float, reduction: str = "none"
) -> torch.Tensor:
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if not weight is None:
        loss = loss * weight[:,None]
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss