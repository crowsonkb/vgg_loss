#!/usr/bin/env python3

"""Reconstruction of a target image from the VGG perceptual loss."""

import torch
from torch import optim
from torchvision import io as tio
from torchvision.transforms import functional as TF

import vgg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    crit_vgg = vgg_loss.VGGLoss().to(device)
    crit_tv = vgg_loss.TVLoss(p=2)

    target = tio.read_image('DSC00261.jpg')[None] / 255
    target = TF.resize(target, (256, 256), 3).to(device)
    target_act = crit_vgg.get_features(target)

    input = torch.rand_like(target) / 255 + 0.5
    input.requires_grad_(True)

    opt = optim.Adam([input], lr=0.025)

    try:
        for i in range(1000):
            opt.zero_grad()
            loss = crit_vgg(input, target_act, target_is_features=True)
            loss += crit_tv(input) * 20
            print(i, loss.item())
            loss.backward()
            opt.step()
    except KeyboardInterrupt:
        pass

    TF.to_pil_image(input[0].clamp(0, 1)).save('out.png')


if __name__ == '__main__':
    main()
