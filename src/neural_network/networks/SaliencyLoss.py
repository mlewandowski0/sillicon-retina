import torch
import numpy as np 


class SaliencyLoss(torch.nn.Module):
    def __init__(self, eps=1e-7): # Smaller eps for precision
        super(SaliencyLoss, self).__init__()
        self.eps = eps
    
    def _flatten(self, x):
        return x.view(x.size(0), -1)

    def _densify(self, x):
        # x: (B, 1, H, W)
        x = x.clamp_min(0.0)
        x = self._flatten(x)
        # Add eps to sum to prevent division by zero, but keep x sparse if it is 0
        denom = x.sum(dim=1, keepdim=True) + self.eps 
        return x / denom
    
    def _kl(self, y, q):
        # y, q: (B, N) normalized probability distributions
        # Add eps inside log to prevent log(0) = -inf
        y = y + self.eps
        q = q + self.eps
        
        # KL = sum( q * log(q/y) ) = sum( q * (log q - log y) )
        log_q = torch.log(q)
        log_y = torch.log(y)
        kl = (q * (log_q - log_y)).sum(dim=1)
        return kl.mean()

    def _cc(self, y, q):
        # Pearson correlation
        # y, q: (B, N)
        y_mu = y.mean(dim=1, keepdim=True)
        q_mu = q.mean(dim=1, keepdim=True)
        
        y_c = y - y_mu
        q_c = q - q_mu
        
        # Mean of product
        num = (y_c * q_c).mean(dim=1)
        
        # Variances
        y_var = y_c.pow(2).mean(dim=1)
        q_var = q_c.pow(2).mean(dim=1)
        
        # Denominator: sqrt(var + eps)
        # Putting eps INSIDE sqrt prevents infinite gradient at 0 variance
        den = torch.sqrt(y_var + self.eps) * torch.sqrt(q_var + self.eps)
        
        cc = num / den
        return -cc.mean()

    def forward(self, predicted_continuous, target_continuous, alpha_cc=0.1, normalize=True):
        if normalize:
            y = self._densify(predicted_continuous)
            q = self._densify(target_continuous)
        else:
            y = self._flatten(predicted_continuous)
            q = self._flatten(target_continuous)
            
        kl_loss = self._kl(y, q)
        cc_loss = self._cc(y, q)
        
        total_loss = kl_loss + alpha_cc * cc_loss
        return total_loss, kl_loss, cc_loss