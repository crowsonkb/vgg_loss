import torch
from torch import nn, optim
from torchvision import io as tio
from torchvision.transforms import functional as TF

import vgg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    crit_vgg = vgg_loss.VGGLoss().to(device)
    crit_l2 = nn.MSELoss()
    crit_tv = vgg_loss.TVLoss(p=1)

    target = tio.read_image('DSC00261.jpg')[None] / 255
    target = TF.resize(target, (256, 256), 3)
    target += (torch.rand_like(target) - 0.5) / 4
    target = target.clamp(0, 1)
    TF.to_pil_image(target[0]).save('target.png')
    target = target.to(device)
    target_act = crit_vgg.get_features(target)

    input = target.clone()
    input.requires_grad_(True)

    opt = optim.Adam([input], lr=0.01)

    try:
        for i in range(250):
            opt.zero_grad()
            loss = crit_vgg(input, target_act, target_is_features=True)
            loss += crit_l2(input, target) * 1500
            loss += crit_tv(input) * 250
            print(i, loss.item())
            loss.backward()
            opt.step()
    except KeyboardInterrupt:
        pass

    TF.to_pil_image(input[0].clamp(0, 1)).save('out.png')


if __name__ == '__main__':
    main()
