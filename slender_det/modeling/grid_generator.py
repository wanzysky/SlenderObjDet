import torch

from concern.support import make_dual


def zero_center_grid(size):

    """
    Generate coordinate zero-centered grid.
    """

    def valid_size(s):
        assert s % 2 == 1, "size should be odd integer."

    H, W = make_dual(size)
    valid_size(H)
    valid_size(W)
    h = (H - 1) // 2
    w = (W - 1) // 2
    
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-h, h, H), 
        torch.linspace(-w, w, W)
    )
    # h, w, 2
    return torch.stack([grid_x, grid_y], axis=2)


def uniform_grid(size):
    """
    Generate uniform coordinate grid.
    """
    H, W = make_dual(size)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, H), 
        torch.linspace(0, W - 1, W)
    )
    return torch.stack([grid_x, grid_y], axis=2)
