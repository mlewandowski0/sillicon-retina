import torch 
import torch.nn as nn
import numpy as np 
from pathlib import Path
import time 
from tqdm import tqdm
import wandb
import torch.nn.functional as F

class E2VIDModel(nn.Module):
    """ Modified E2VID model for ECOCO dataset reconstruction

    """

    def __init__(self, num_bins=5, num_prev_images=3, dropout_rate=0.2):
        super().__init__()
        self.num_bins = num_bins
        self.num_prev_images = num_prev_images
        self.dropout = nn.Dropout2d(dropout_rate)

        # Input channels: B temporal bins + K previous images
        input_channels = num_bins + num_prev_images

        # Encoder layers (downsampling)
        self.encoder1 = self._conv_layer(input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.encoder2 = self._conv_layer(64, 128, kernel_size=5, stride=2, padding=2)
        self.encoder3 = self._conv_layer(128, 256, kernel_size=5, stride=2, padding=2)
        self.encoder4 = self._conv_layer(256, 512, kernel_size=5, stride=2, padding=2)

        # Residual blocks
        self.resblock1 = self._residual_block(512)
        self.resblock2 = self._residual_block(512)

        # Decoder layers (upsampling) with skip connections
        self.decoder1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2)
        self.decoder1_conv = self._conv_layer(256 + 256, 256, kernel_size=3, padding=1)

        self.decoder2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2)
        self.decoder2_conv = self._conv_layer(128 + 128, 128, kernel_size=3, padding=1)

        self.decoder3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2)
        self.decoder3_conv = self._conv_layer(64 + 64, 64, kernel_size=3, padding=1)

        self.decoder4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2)
        self.decoder4_conv = self._conv_layer(32 + input_channels, 32, kernel_size=3, padding=1)

        # Final prediction layer
        self.pred = nn.Conv2d(32, 1, kernel_size=1)

    def _conv_layer(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, event_voxel, prev_images):
        """
        Args:
            event_voxel: (B, num_bins, H, W) - Event voxel grid
            prev_images: (B, num_prev_images, H, W) - Previous reconstructed images

        Returns:
            reconstructed_image: (B, 1, H, W) - Reconstructed grayscale image
        """
        # Concatenate event tensor and previous images
        x = torch.cat([event_voxel, prev_images], dim=1)
        skip0 = x

        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # Residual blocks
        res = self.resblock1(x4)
        x = res + x4
        x = self.dropout(x)

        res = self.resblock2(x)
        x = res + x
        x = self.dropout(x)

        # Decoder with skip connections
        x = F.relu(self.decoder1(x))
        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder1_conv(x)

        x = F.relu(self.decoder2(x))
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2_conv(x)

        x = F.relu(self.decoder3(x))
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder3_conv(x)

        x = F.relu(self.decoder4(x))
        x = F.interpolate(x, size=skip0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip0], dim=1)
        x = self.decoder4_conv(x)

        # Final prediction with sigmoid activation
        x = self.pred(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    
    images, voxels = 3, 5 
    batch_size = 2
    model = E2VIDModel(num_bins=voxels, num_prev_images=images)
    
    event_voxel = torch.randn(batch_size, voxels, 180, 240)
    prev_images = torch.randn(batch_size, images, 180, 240)
    
    out = model(event_voxel, prev_images)
    print("Output shape: ", out.shape)  # Should be (batch_size, 1, 64, 64)
    