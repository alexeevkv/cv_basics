import torch
from segmentation_models_pytorch.utils.losses import DiceLoss


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, min=1e-6, max=1 - 1e-6)
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
            weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))
