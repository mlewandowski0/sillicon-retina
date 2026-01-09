import torch
import numpy as np 

class SaliencyLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):  # Increased eps
        super(SaliencyLoss, self).__init__()
        self.eps = eps

    def _flatten(self, x):
        return x.view(x.size(0), -1)

    def _densify(self, x):
        x = x.clamp_min(0.0)
        x = self._flatten(x)
        denom = x.sum(dim=1, keepdim=True)
        
        # Handle case where sum is zero or near-zero
        # Return uniform distribution instead
        is_zero = denom < self.eps
        denom = denom.clamp_min(self.eps)
        result = x / denom
        
        # For samples with zero sum, replace with uniform
        if is_zero.any():
            uniform = torch.ones_like(result) / result.size(1)
            result = torch.where(is_zero.expand_as(result), uniform, result)
        
        return result

    def _kl(self, y, q):
        # Add small constant before log to avoid log(0)
        y_safe = y + self.eps
        q_safe = q + self.eps
        
        # Renormalize after adding eps
        y_safe = y_safe / y_safe.sum(dim=1, keepdim=True)
        q_safe = q_safe / q_safe.sum(dim=1, keepdim=True)
        
        kl = (q_safe * (q_safe.log() - y_safe.log())).sum(dim=1)
        
        # Clamp to prevent extreme values
        kl = kl.clamp(min=0.0, max=100.0)
        
        return kl.mean()

    def _cc(self, y, q):
        y_mu = y.mean(dim=1, keepdim=True)
        q_mu = q.mean(dim=1, keepdim=True)
        y_c = y - y_mu
        q_c = q - q_mu
        
        y_var = y_c.pow(2).mean(dim=1)
        q_var = q_c.pow(2).mean(dim=1)
        
        # Check for near-zero variance (constant signal)
        y_std = y_var.sqrt()
        q_std = q_var.sqrt()
        
        # Mask samples with near-zero variance
        valid_mask = (y_std > self.eps) & (q_std > self.eps)
        
        num = (y_c * q_c).mean(dim=1)
        den = (y_std * q_std).clamp_min(self.eps)
        
        cc = num / den
        
        # For invalid samples (near-constant), use 0 correlation
        cc = torch.where(valid_mask, cc, torch.zeros_like(cc))
        
        # Clamp CC to valid range
        cc = cc.clamp(-1.0, 1.0)
        
        return -cc.mean()

    def forward(self, predicted_continuous, target_continuous, alpha_cc=0.1, normalize=True):
        # Check for NaN/Inf inputs
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
        
        # Final NaN check
        if torch.isnan(total_loss):
            return torch.tensor(0.0, device=predicted_continuous.device, requires_grad=True), \
                   torch.tensor(0.0, device=predicted_continuous.device), \
                   torch.tensor(0.0, device=predicted_continuous.device)
        
        return total_loss, kl_loss, cc_loss