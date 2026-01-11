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
        denom = x.sum(dim = 1, keepdim=True).clamp_min(self.eps)        
        return x / denom
    
    def _kl(self, y, q):
        # y, q are now (B, N) and are non-negative and (ideally) normalized to 
        y = y.clamp_min(self.eps)
        q = q.clamp_min(self.eps)
        kl = (q * (q.log() - y.log())).sum(dim=1)
        return kl.mean()

    def _cc(self, y, q):
        # Pearson correlation, then negate ( since we minimize loss)
        # y, q are now (B, N)
        y_mu = y.mean(dim=1, keepdim=True)
        q_mu = q.mean(dim=1, keepdim=True)
        y_c = y - y_mu
        q_c = q - q_mu
        num = (y_c * q_c).mean(dim=1)
        den = (y_c.pow(2).mean(dim=1).sqrt().clamp_min(self.eps) *
        q_c.pow(2).mean(dim=1).sqrt().clamp_min(self.eps))
        cc = num / den 
        return -cc.mean()

    def forward(self, predicted_continuous, target_continuous, alpha_cc=0.1, normalize=True):
        """ 
                    predicted_continuous: (B, 1, H, W) - raw output of the network 
                    target_continuous: (B, 1, H, W) - ground truth saliency map    
                """
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