import lpips
import torch 
import numpy as np 
import torch.nn as nn

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self, alpha = 0.1, use_lpips=True):
        super(PerceptualLoss, self).__init__()
        self.use_lpips = use_lpips
        if use_lpips:
            self.lpips_loss = lpips.LPIPS(net='vgg').cuda()
    
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        
    def forward(self, pred, target):
        if self.use_lpips:
            # Compute LPIPS loss
            pred_rgb = pred.repeat(1, 3, 1, 1)
            target_rgb = target.repeat(1, 3, 1, 1)
            lpips_loss = self.lpips_loss(pred_rgb, target_rgb).mean()
            l1_loss = self.l1_loss(pred, target)
            return lpips_loss + self.alpha * l1_loss
        else:
            # Compute L1 loss
            return self.l1_loss(pred, target) + self.alpha * self.mse_loss(pred, target)    
    