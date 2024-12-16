import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


def get_sample_feat(left_fea, right_fea, disp, radius):

    device = disp.device

    b, c, h, w = right_fea.shape

    w_grid = torch.arange(0, w, device=device).view(1, 1, 1, -1).repeat(b, 2*radius+1, h, 1)
    disp_range = torch.stack([disp + i for i in range(-radius, radius+1)], dim=-1).squeeze(1).permute(0, 3, 1, 2).contiguous()
    disp_range = torch.clamp(disp_range, min=0.)

    w_index = w_grid - disp_range
    h_index = torch.arange(0, h, device=device).view(1, 1, -1, 1).repeat(b, 2*radius+1, 1, w).contiguous()

    h_normalize = 2. * h_index/(h - 1) - 1
    w_normalize = 2. * w_index/(w - 1) - 1

    grid = torch.stack((w_normalize, h_normalize), dim=-1).contiguous().view(b, -1, w, 2)

    grid = grid.type(right_fea.type())
    neigFeatL = F.grid_sample(right_fea, grid, align_corners=True).view(b, c, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()

    channel_per_group = c // 16
    ret = (left_fea.unsqueeze(1).repeat(1, 2*radius+1, 1, 1, 1).contiguous() * neigFeatL).view(b, 2*radius++1, channel_per_group, 16, h, w).mean(dim=2)

    return ret.contiguous().view(b, -1, h, w)


class GlobalPerception:
    def __init__(self, init_fmap1, init_fmap2, num_levels=2, radius=4):


        self.num_levels = num_levels
        self.radius = radius
        self.init_corr_pyramid = []

        init_corr = GlobalPerception.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        init_corr = init_corr.reshape(b * h * w, 1, 1, w2)

        self.init_corr_pyramid.append(init_corr)

        for i in range(self.num_levels - 1):
            init_corr = F.avg_pool2d(init_corr, [1, 2], stride=[1, 2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords, generate_mode=False):
        r = 32
        b, _, h, w = disp.shape
        coords = torch.arange(w, device=disp.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        out_pyramid = []
        for i in range(self.num_levels):
            dx = torch.linspace(-r, r, 2 * r + 1, device="cuda")
            dx = dx.view(1, 1, 2 * r + 1, 1)
            x0 = dx + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b * h * w, 1, 1, 1) / 2 ** i - disp.reshape(b * h * w, 1, 1, 1) / 2 ** i + dx
            init_coords_lvl = torch.cat([init_x0, y0], dim=-1).contiguous()
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, generate_mode=generate_mode)
            init_corr = init_corr.view(b, h, w, -1)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def mask_gen(b, h, w):
        pos_r = torch.linspace(0, w - 1, w, device="cuda")[None, None, None, :]
        pos_l = torch.linspace(0, w - 1, w, device="cuda")[None, None, :, None]
        pos = pos_l - pos_r
        pos = torch.where(pos>0, torch.ones_like(pos), torch.zeros_like(pos)).repeat(b,h,1,1).contiguous()
        return pos

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)

        mask = GlobalPerception.mask_gen(B, H, W1)
        corr = corr * mask
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr