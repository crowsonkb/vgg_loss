"""Helper functions for the VGG perceptual loss."""

import torch


def inspect_outputs(module):
    """Registers hooks on each submodule that print their outputs."""

    def make_hook(name):
        return lambda m, i, o: print(f'({name}) {type(m).__name__}: {o}')

    for name, mod in module.named_children():
        mod.register_forward_hook(make_hook(name))


def batchify_image(input):
    """Promotes the input tensor (an image or a batch of images) to a 4D tensor
    with three channels, if it is not already. Strips alpha channels if
    present.
    """
    if input.ndim == 2:
        input = input[None]
    if input.ndim == 3:
        input = input[None]
    if input.ndim != 4:
        raise ValueError('input.ndim must be 2, 3, or 4')
    if input.shape[1] == 2 or input.shape[1] == 4:
        input = input[:, :-1]
    if input.shape[1] == 1:
        input = torch.cat([input] * 3, dim=1)
    if input.shape[1] != 3:
        raise ValueError('input must have 1-4 channels')
    return input
