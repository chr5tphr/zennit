'''Functions to produce a canonical form of models fit for LRP'''
import torch


def merge_batch_norm(module, batch_norm):
    '''Update parameters of a linear layer to additionally include a Batch Normalization operation and update the batch
    normalization layer to instead compute the identity.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        Linear layer with mandatory attributes `weight` and `bias`.
    batch_norm: obj:`torch.nn.Module`
        Batch Normalization module with mandatory attributes `running_mean`, `running_var`, `weight`, `bias` and `eps`
    '''
    original_weight = module.weight.data
    if module.bias is None:
        module.bias = torch.nn.Parameter(torch.zeros(1, device=original_weight.device, dtype=original_weight.dtype))
    original_bias = module.bias.data

    denominator = (batch_norm.running_var + batch_norm.eps) ** .5
    scale = (batch_norm.weight / denominator)

    # merge batch_norm into linear layer
    module.weight.data = (original_weight * scale[:, None, None, None])
    module.bias.data = (original_bias - batch_norm.running_mean) * scale + batch_norm.bias

    # change batch_norm parameters to produce identity
    batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
    batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
    batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
    batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)
