import torch
from torch import nn
from torchvision import models, transforms


class TVLoss(nn.Module):
    """Total variation loss (Lp penalty on image gradient magnitude).

    The input must be at least 2D and at least 2x2. Multichannel images and
    batches are supported.

    ``p=1`` yields the originally proposed (isotropic) 2D total variation
    norm (see https://en.wikipedia.org/wiki/Total_variation_denoising).
    ``p=2`` yields a variant that is often used for smoothing out noise in
    reconstructions of images from neural network feature maps.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``mean``.
    """

    def __init__(self, p, reduction='mean'):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input):
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = (x_diff**2 + y_diff**2 + 1e-8)**(self.p / 2)
        if self.reduction == 'none':
            return diff
        out = torch.sum(diff)
        if self.reduction == 'mean':
            out /= input.numel()
        return out


class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1. H and W must be at least 2.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``mean``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = models.vgg16(pretrained=True).features[:9]
        self.model.eval()
        self.model.requires_grad_(False)
        self.loss = nn.MSELoss(reduction=reduction)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return self.loss(input_feats, target_feats)