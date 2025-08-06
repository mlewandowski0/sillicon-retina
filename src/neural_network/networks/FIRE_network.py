import torch.nn as nn
import torch.nn.functional as F
import torch
from .convGRU import ConvGRUCell
from math import ceil

class Head(nn.Module):
    def __init__(self, input, out_channels, ks1, add_bn=True):
        super().__init__()

        padding = ks1 // 2
        self.conv1 = nn.Conv2d(input, out_channels, kernel_size=ks1, padding=padding)

        if add_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Identity()

        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinearity(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, add_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=k // 2)
        if add_bn:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        v = self.conv(x)
        v = self.bn(v)
        v = self.act(v)
        return x + v


class FIRE_network(nn.Module):
    def __init__(self, in_ch, hid1=16, ks1=5, add_bn=True, out_sigmoid=True):
        super().__init__()

        self.in_ch = in_ch
        self.hid1 = hid1
        self.ks1 = ks1

        self.head = Head(in_ch, hid1, ks1, add_bn)

        self.convGRU1 = ConvGRUCell(hid1, hid1, ks1)
        self.nonlinearity1 = nn.ReLU()
        self.residual1 = ResidualBlock(hid1, hid1, ks1, add_bn=add_bn)

        self.convGRU2 = ConvGRUCell(hid1, hid1, ks1)
        self.nonlinearity2 = nn.ReLU()
        self.residual2 = ResidualBlock(hid1, hid1, ks1, add_bn=add_bn)

        self.conv = nn.Conv2d(hid1, 1, kernel_size=1, padding=0)

        if out_sigmoid:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Identity()


        self.state1 = None
        self.state2 = None


    def _reset_hidden(self, input_shape, batch_size, device, dtype=torch.float32):
        H, W = input_shape
        h = torch.zeros((batch_size, self.hid1, H, W), device=device, dtype=dtype)
        return h

    def reset(self, input_shape, batch_size):
        # set hidden states to the model's device/dtype
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        self.state1 = self._reset_hidden(input_shape, batch_size, device=dev, dtype=dt)
        self.state2 = self._reset_hidden(input_shape, batch_size, device=dev, dtype=dt)


    def forward(self, x):

        x = self.head(x)

        x = self.convGRU1(x, self.state1)
        self.state1 = x
        x = self.nonlinearity1(x)

        x = self.residual1(x)

        x = self.convGRU2(x, self.state2)
        self.state2 = x
        x = self.nonlinearity2(x)


        x = self.residual2(x)

        x = self.conv(x)
        x = self.out(x)
        return x

if __name__ == "__main__":
    batch_size = 2
    input_channels = 5
    resolution = (180, 240)

    model = FIRE_network(in_ch=input_channels, hid1=16, ks1=3, add_bn=False, out_sigmoid=False)
    model.reset(input_shape=resolution, batch_size=batch_size)

    for _ in range(5):
        batch = torch.zeros(batch_size, input_channels, resolution[0], resolution[1])
        s = model(batch)

    print(s.shape)
