'''Additional Utility Layers'''
import torch


class Sum(torch.nn.Module):
    '''Compute the sum along an axis.

    Parameters
    ----------
    dim : int
        Dimension over which to sum.
    '''
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        '''Computes the sum along a dimension.'''
        return torch.sum(input, dim=self.dim)
