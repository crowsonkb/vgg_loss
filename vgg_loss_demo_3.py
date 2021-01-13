"""An autoencoder using the VGG perceptual loss."""

import torch
from torch import optim, nn
from torch.utils import data
from torchvision import datasets, transforms

import vgg_loss

BATCH_SIZE = 100
EPOCHS = 100


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf = transforms.ToTensor()
    train_set = datasets.CIFAR10('data/cifar10', download=True, transform=tf)
    train_dl = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               pin_memory=True)
    test_set = datasets.CIFAR10('data/cifar10', train=False, transform=tf)
    test_dl = data.DataLoader(test_set, batch_size=BATCH_SIZE,
                              pin_memory=True)

    encoder = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        nn.AvgPool2d(2, ceil_mode=True),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AvgPool2d(2, ceil_mode=True),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.AvgPool2d(2, ceil_mode=True),
        nn.Flatten(),
        nn.Linear(1024, 128), nn.Tanh(),
    ).to(device)

    decoder = nn.Sequential(
        nn.Linear(128, 1024), nn.ReLU(),
        nn.Unflatten(-1, (64, 4, 4)),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid(),
    ).to(device)

    model = nn.Sequential(encoder, decoder)
    print('Parameters:', sum(map(lambda x: x.numel(), model.parameters())))

    # crit = nn.MSELoss()
    crit = vgg_loss.WeightedLoss([vgg_loss.VGGLoss(shift=2),
                                  nn.MSELoss(),
                                  vgg_loss.TVLoss(p=1)],
                                 [1, 40, 10]).to(device)
    # helpers.inspect_outputs(crit)
    opt = optim.Adam(model.parameters())
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5,
                                                 verbose=True)

    def train():
        model.train()
        crit.train()
        i = 0
        for batch, _ in train_dl:
            i += 1
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(batch)
            loss = crit(out, batch)
            if i % 50 == 0:
                print(i, loss.item())
            loss.backward()
            opt.step()

    @torch.no_grad()
    def test():
        model.eval()
        crit.eval()
        losses = []
        for batch, _ in test_dl:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            losses.append(crit(out, batch))
        loss = sum(losses) / len(losses)
        print('Validation loss:', loss.item())
        sched.step(loss)

    @torch.no_grad()
    def demo():
        model.eval()
        batch = torch.cat([test_set[i][0][None] for i in range(10)])
        out = model(batch.to(device)).cpu()
        col_l = torch.cat(list(batch), dim=1)
        col_r = torch.cat(list(out), dim=1)
        grid = torch.cat([col_l, col_r], dim=2)
        transforms.functional.to_pil_image(grid).save('demo.png')
        print('Wrote example grid to demo.png.')

    try:
        for epoch in range(EPOCHS):
            print('Epoch', epoch + 1)
            train()
            test()
            demo()
    except KeyboardInterrupt:
        pass

    torch.save(model.state_dict(), 'autoencoder.pth')
    print('Wrote trained model to autoencoder.pth.')


if __name__ == '__main__':
    main()
