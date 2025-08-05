import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from .temporalConv import ConvLSTMCell


class Head(nn.Module):
    def __init__(self, input_chs, output_chs, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(input_chs, output_chs, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(output_chs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Conv -> BN -> ReLU
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownsampleRecurrentBlock(nn.Module):
    def __init__(self, input_chs, convlstm_in_ch, out_channels, k_down=5, k_lstm=3):
        super().__init__()
        self.out_channels = out_channels

        # strided conv downsample + BN + ReLU
        self.down = nn.Conv2d(input_chs, convlstm_in_ch, kernel_size=k_down, stride=2, padding=k_down // 2)
        self.down_bn = nn.BatchNorm2d(convlstm_in_ch)
        self.down_relu = nn.ReLU(inplace=True)

        # ConvLSTM (in: convlstm_in_ch, out: out_channels)
        self.convlstm = ConvLSTMCell(convlstm_in_ch, out_channels, ks=k_lstm, bias=True, peephole=True)

        # (optional) post-LSTM BN+ReLU to stabilize
        self.post_bn = nn.BatchNorm2d(out_channels)
        self.post_relu = nn.ReLU(inplace=True)

        self.state = None  # (h, c)

    def forward(self, x):
        if self.state is None:
            raise RuntimeError("Hidden state is None. Call model.reset(...) before forward().")

        x = self.down(x)
        x = self.down_bn(x)
        x = self.down_relu(x)

        h, c = self.convlstm(x, self.state)
        h = self.post_bn(h)
        h = self.post_relu(h)
        self.state = (h, c)
        return h

    def reset(self, input_shape, batch_size, device=None, dtype=None):
        H, W = input_shape
        h = torch.zeros((batch_size, self.out_channels, H, W), device=device, dtype=dtype)
        c = torch.zeros_like(h)
        self.state = (h, c)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=k // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        v = self.conv(x)
        v = self.bn(v)
        v = self.act(v)
        return x + v


class Upsample(nn.Module):
    """
    Upsample -> fuse with skip (cat or sum) -> Conv(k) -> BN -> ReLU

    Args:
        in_ch:   channels of the decoder input BEFORE upsampling
        skip_ch: channels of the skip tensor
        out_ch:  output channels of the block
        k:       conv kernel size after fusion
        skip_type: 'cat' or 'sum'
                  - 'cat': conv input channels = in_ch + skip_ch
                  - 'sum': conv input channels = max(in_ch, skip_ch); will 1x1-project if needed
    """
    def __init__(self, in_ch, skip_ch, out_ch, k=5, skip_type="cat"):
        super().__init__()
        self.skip_type = skip_type

        # Optional projection if using sum and channels don't match
        self.proj = None
        if skip_type == "sum" and in_ch != skip_ch:
            self.proj = nn.Conv2d(in_ch, skip_ch, kernel_size=1)
            conv_in = skip_ch
        elif skip_type == "sum":
            conv_in = in_ch  # == skip_ch
        elif skip_type == "cat":
            conv_in = in_ch + skip_ch
        else:
            raise NotImplementedError(f"skip_type {skip_type} not supported (use 'cat' or 'sum').")

        self.fuse_conv = nn.Conv2d(conv_in, out_ch, kernel_size=k, padding=k // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # upsample first
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # fuse
        if self.skip_type == "sum":
            if self.proj is not None:
                x = self.proj(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        else:  # 'cat'
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        # refine
        x = self.fuse_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class E2VID_convLSTM(nn.Module):
    def __init__(self, input_channels=5, skip_type="cat"):
        super().__init__()
        assert skip_type in ("cat", "sum")

        # Encoder head
        self.head = Head(input_chs=input_channels, output_chs=32, kernel_size=5, stride=1)

        # Encoder (downsample + ConvLSTM)
        self.recurrent = nn.ModuleList([
            DownsampleRecurrentBlock(32, 64, 64, k_down=5, k_lstm=3),    # H/2, 64
            DownsampleRecurrentBlock(64, 128, 128, k_down=5, k_lstm=3),  # H/4, 128
            DownsampleRecurrentBlock(128, 256, 256, k_down=5, k_lstm=3), # H/8, 256
        ])

        # Bottleneck residuals
        self.residuals = nn.ModuleList([
            ResidualBlock(256, 256, k=3),
            ResidualBlock(256, 256, k=3),
        ])

        # Decoder (upsample → fuse with skip → conv)
        # Shapes: x:256 with skip3:128 -> 128; x:128 with skip2:64 -> 64; x:64 with skip1:32 -> 32
        self.upsamples = nn.ModuleList([
            Upsample(in_ch=256, skip_ch=128, out_ch=128, k=5, skip_type=skip_type),
            Upsample(in_ch=128, skip_ch=64,  out_ch=64,  k=5, skip_type=skip_type),
            Upsample(in_ch=64,  skip_ch=32,  out_ch=32,  k=5, skip_type=skip_type),
        ])

        # Final prediction conv (after an extra fuse with head via concat)
        self.pred = nn.Conv2d(32 + 32, 1, kernel_size=5, stride=1, padding=2)

    def reset(self, input_shape, batch_size):
        # set hidden states to the model's device/dtype
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        for i, block in enumerate(self.recurrent):
            div = 2 ** (i + 1)
            H = ceil(input_shape[0] / div)
            W = ceil(input_shape[1] / div)
            block.reset((H, W), batch_size, device=dev, dtype=dt)

    def forward(self, x):
        # Encoder
        skip1 = self.head(x)                 # H,   32 ch
        skip2 = self.recurrent[0](skip1)     # H/2, 64 ch
        skip3 = self.recurrent[1](skip2)     # H/4, 128 ch
        skip4 = self.recurrent[2](skip3)     # H/8, 256 ch

        # Bottleneck
        x = skip4
        for layer in self.residuals:
            x = layer(x)

        # Decoder (start with skip3; no extra resizes needed)
        x = self.upsamples[0](x, skip3)      # -> H/4, 128 ch
        x = self.upsamples[1](x, skip2)      # -> H/2, 64 ch
        x = self.upsamples[2](x, skip1)      # -> H,   32 ch

        # Optional extra fuse with head (concat like your original design)
        x = torch.cat([x, skip1], dim=1)     # 32 + 32 = 64 ch

        # Prediction + sigmoid
        x = self.pred(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    def check_cat_version(batch_size, input_channels, resolution):
        model = E2VID_convLSTM(input_channels=input_channels, skip_type="cat")
        model.reset(input_shape=resolution, batch_size=batch_size)

        for _ in range(5):
            batch = torch.zeros(batch_size, input_channels, resolution[0], resolution[1])
            s = model(batch)

        print(s.shape)

    def check_sum_version(batch_size, input_channels, resolution):
        model = E2VID_convLSTM(input_channels=input_channels, skip_type="sum")
        model.reset(input_shape=resolution, batch_size=batch_size)

        for _ in range(5):
            batch = torch.zeros(batch_size, input_channels, resolution[0], resolution[1])
            s = model(batch)

        print(s.shape)


    batch_size = 2
    input_channels = 5
    resolution = (180, 240)

    check_cat_version(batch_size, input_channels, resolution)
    check_sum_version(batch_size, input_channels, resolution)