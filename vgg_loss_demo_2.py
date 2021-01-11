import torch
from torch import optim
from torchvision import io as tio
from torchvision.transforms import functional as TF

import vgg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crit = vgg_loss.VGGLoss().to(device)
    crit_tv = vgg_loss.TVLoss(p=1)
    target = tio.read_image('DSC00261.jpg')[None] / 255
    target = TF.resize(target, (256, 256), 3)
    target += (torch.rand_like(target) - 0.5) / 4
    target = target.clamp(0, 1)
    TF.to_pil_image(target[0]).save('target.png')
    target = target.to(device)
    target_act = crit.get_features(target)
    input = target.clone()
    input.requires_grad_(True)
    opt = optim.Adam([input], lr=0.01)

    try:
        for i in range(1000):
            opt.zero_grad()
            loss = crit(input, target_act, target_is_features=True)
            loss += crit_tv(input) * 50
            print(i, loss.item())
            loss.backward()
            opt.step()
    except KeyboardInterrupt:
        pass

    TF.to_pil_image(input[0].clamp(0, 1)).save('out.png')


if __name__ == '__main__':
    main()
