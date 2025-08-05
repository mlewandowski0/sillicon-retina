import torch.nn as nn
import torch.nn.functional as F
import torch

class TemporalConsistencyLoss(nn.Module):
    def __init__(self, alpha=50.0, lambda_tc = 5.0, ignore_first = 2):
        super(TemporalConsistencyLoss, self).__init__()

        self.alpha = alpha
        self.lambda_tc = lambda_tc
        self.ignore_first = ignore_first

    @torch.no_grad()
    def _warp(self, img, flow):
        # img: (B, 1, H, W), flow (B, 2, H, W) in pixels, grid_sample expects [-1, 1] norm coords
        B, _, H, W = img.shape

        # build normalized base grid
        yy, xx = torch.meshgrid(torch.arange(H, device=img.device),
                                torch.arange(W, device=img.device), indexing='ij')

        xx = xx.unsqueeze(0).expand(B, -1, -1).float() + flow[:, 0]
        yy = yy.unsqueeze(0).expand(B, -1, -1).float() + flow[:, 1]

        gx = 2.0 * (xx / max(W - 1, 1)) - 1.0
        gy = 2.0 * (yy / max(H - 1, 1)) - 1.0

        grid = torch.stack([gx, gy], dim=-1)

        return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, flow, Ihat, Ihat_prev, Igt, Igt_prev):
        # each arg: (B, 1, H, W)
        if flow is None:
            return torch.tensor(0., device=Ihat.device)

        with torch.no_grad():
            Igt_prev_w = self._warp(Igt_prev, flow)
            M = torch.exp(-self.alpha * (Igt - Igt_prev_w).pow(2).sum(dim=1, keepdim=True))
            Ihat_prev_w = self._warp(Ihat_prev, flow)

        return (M * (Ihat - Ihat_prev_w).abs()).mean() * self.lambda_tc