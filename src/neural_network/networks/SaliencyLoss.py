import torch
import numpy as np 

class SaliencyLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(SaliencyLoss, self).__init__()
        self.eps = eps

    def _flatten(self, x):
        # (B, 1, H, W) -> (B, H * W)
        return x.view(x.size(0), -1)

    def _densify(self, x):
        # take the saliency map represented as (B, 1, H, W) and densify it to make sure it sums to 1
        x = x.clamp_min(0.0)
        x = self._flatten(x)
        denom = x.sum(dim=1, keepdim=True)
        
        # Check if any sample has zero sum - this causes NaN
        has_zero_sum = (denom < self.eps).any()
        denom = denom.clamp_min(self.eps)
        result = x / denom
        
        # Only for samples with zero sum, add small uniform noise to avoid degenerate distribution
        if has_zero_sum:
            is_zero = (x.sum(dim=1, keepdim=True) < self.eps)
            if is_zero.any():
                # Add tiny uniform component only to zero-sum samples
                uniform = torch.ones_like(result) / result.size(1)
                result = torch.where(is_zero.expand_as(result), uniform, result)
        
        return result

    def _kl(self, y, q):
        # Original logic preserved
        y = y.clamp_min(self.eps)
        q = q.clamp_min(self.eps)
        kl = (q * (q.log() - y.log())).sum(dim=1)
        return kl.mean()

    def _cc(self, y, q):
        # Pearson correlation, then negate (since we minimize loss)
        y_mu = y.mean(dim=1, keepdim=True)
        q_mu = q.mean(dim=1, keepdim=True)
        y_c = y - y_mu
        q_c = q - q_mu
        
        num = (y_c * q_c).mean(dim=1)
        
        y_std = y_c.pow(2).mean(dim=1).sqrt()
        q_std = q_c.pow(2).mean(dim=1).sqrt()
        
        den = y_std * q_std
        
        # This is the key fix: detect near-zero variance samples
        valid_mask = den > self.eps
        
        # Safe division
        den_safe = den.clamp_min(self.eps)
        cc = num / den_safe
        
        # For invalid samples (near-constant distributions), set cc to 0
        # This avoids NaN while having minimal impact on the loss
        cc = torch.where(valid_mask, cc, torch.zeros_like(cc))
        
        return -cc.mean()

    def forward(self, predicted_continuous, target_continuous, alpha_cc=0.1, normalize=True):
        """ 
        predicted_continuous: (B, 1, H, W) - raw output of the network 
        target_continuous: (B, 1, H, W) - ground truth saliency map    
        """
        # Early NaN/Inf check - return zero loss to skip this sample
        if torch.isnan(predicted_continuous).any() or torch.isinf(predicted_continuous).any():
            return torch.tensor(0.0, device=predicted_continuous.device, requires_grad=True), \
                   torch.tensor(0.0, device=predicted_continuous.device), \
                   torch.tensor(0.0, device=predicted_continuous.device)
        
        if normalize:
            y = self._densify(predicted_continuous)
            q = self._densify(target_continuous)
        else:
            y = self._flatten(predicted_continuous)
            q = self._flatten(target_continuous)

        kl_loss = self._kl(y, q)
        cc_loss = self._cc(y, q)

        total_loss = kl_loss + alpha_cc * cc_loss
        
        # Final safety check
        if torch.isnan(total_loss):
            return torch.tensor(0.0, device=predicted_continuous.device, requires_grad=True), \
                   torch.tensor(0.0, device=predicted_continuous.device), \
                   torch.tensor(0.0, device=predicted_continuous.device)

        return total_loss, kl_loss, cc_loss