import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder, Feature
from core.geometry import GlobalPerception, get_sample_feat
from core.submodule import *
import time
import numpy as np


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))




    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv = self.conv1_up(conv1)

        return conv

class EGLCRStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_edge = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        # self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.regulize = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        self.atten = nn.Sequential(
            nn.Conv2d(130, 130, kernel_size=1, groups=2, bias=True),
            nn.InstanceNorm2d(130),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(130, 130, kernel_size=1, groups=2, bias=True),
            nn.InstanceNorm2d(130),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(130, 130, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm2d(130),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(130, 3, kernel_size=1, stride=1, bias=True))

        self.edge_texConv = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, bias=True)
            )

        self.edge_dispConv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3, stride=2, bias=True),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 128, kernel_size=3, padding=1, stride=2, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, bias=True)
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(128+128, 256, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=True),
            nn.InstanceNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=True),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=True))

        self.edge_head_init = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=True),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=True))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp, spx_pred).unsqueeze(1)
        return up_disp


    def forward(self, image1, image2, iters=12, max_disp=544, generate_mode=False, test_mode=False):

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):

            f_l = self.feature(image1)
            f_r = self.feature(image2)

            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)

            f_l = torch.cat((f_l, stem_4x), 1)
            f_r = torch.cat((f_r, stem_4y), 1)

            F_l = self.desc(self.conv(f_l))
            F_r = self.desc(self.conv(f_r))

            gwc_volume = build_gwc_volume(F_l, F_r, max_disp//4, 8)
            reg_volume = self.regulize(gwc_volume)
            prob = F.softmax(self.classifier(reg_volume).squeeze(1), dim=1)
            init_disp = regression(prob, max_disp//4)

            xspx = self.spx_4(f_l)
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = F.softmax(self.spx(xspx), 1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
            init_disp_up = context_upsample(init_disp * 4., spx_pred).unsqueeze(1)

            f_tex = self.edge_texConv(f_l.detach())
            edge_bd_init = self.edge_head_init(f_tex)

        global_perception = GlobalPerception
        global_fn = global_perception(F_l.float(), F_r.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        disp = init_disp.float()
        edge = edge_bd_init.float()
        disp_preds = []
        edge_preds = []

        for itr in range(iters):
            disp = disp.detach().clamp(min=0., max=disp.shape[-1])
            if (not test_mode) and np.random.rand() < 0.3:
                noise = (2*torch.rand_like(disp)-1.) * torch.bernoulli(torch.ones_like(disp) * 0.3)
                disp += noise
            edge = edge.detach()
            f_loc = get_sample_feat(F_l.float(), F_r.float(), disp, self.args.corr_radius)
            f_global = global_fn(disp, generate_mode)
            w = F.softmax(self.atten(f_global), dim=1)

            with autocast(enabled=self.args.mixed_precision):
                w_g = w[:, 0, :, :].unsqueeze(1).contiguous()
                w_l = w[:, 1, :, :].unsqueeze(1).contiguous()
                f_scale = torch.cat([w_l * f_loc, w_g * f_global], dim=1).contiguous()
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, f_scale, disp, edge, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)

            disp = disp + delta_disp.float()
            disp_up = self.upsample_disp(disp * 4., mask_feat_4.float(), stem_2x.float())

            with autocast(enabled=self.args.mixed_precision):
                sigma = min(itr * 1. / (iters + 1), 0.8)
                disp_norm = (32 * ((disp_up.detach() - disp_up.min()) / (disp_up.max()-disp_up.min()))).contiguous()
                f_disp = self.edge_dispConv(disp_norm)
                edge = self.edge_head(torch.cat(((1-sigma)*(1-w_l.detach()) * f_tex, sigma*w_l.detach() * f_disp), dim=1))
            if test_mode:
                continue
            disp_preds.append(disp_up)
            edge_preds.append(edge.float())

        if test_mode:
            return disp_up, edge
        return init_disp_up, disp_preds, edge_bd_init, edge_preds