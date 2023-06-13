import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import lang_tf_enc, TransformerEncoderLayer, TransformerEncoder
from .position_encoding import PositionEmbeddingSine


class ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        leaky=False,
        relu=True,
        instance=False,
    ):
        super(ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        if instance:
            self.add_module(
                "bn",
                nn.InstanceNorm2d(num_features=out_channels),
            )
        else:
            self.add_module(
                "bn",
                nn.BatchNorm2d(
                    num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
                ),
            )

        if leaky:
            self.add_module("relu", nn.LeakyReLU(0.1))
        elif relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(ConvBatchNormReLU, self).forward(x)


def concat_coord(x):
    ins_feat = x  # [bt, c, h, w] [512, 26, 26]
    batch_size, c, h, w = x.size()

    float_h = float(h)
    float_w = float(w)

    y_range = torch.arange(0., float_h, dtype=torch.float32)
    y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
    x_range = torch.arange(0., float_w, dtype=torch.float32)
    x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
    x_range = x_range[None, :]
    y_range = y_range[:, None]
    x = x_range.repeat(h, 1)
    y = y_range.repeat(1, w)

    x = x[None, None, :, :]
    y = y[None, None, :, :]
    x = x.repeat(batch_size, 1, 1, 1)
    y = y.repeat(batch_size, 1, 1, 1)
    x = x.cuda()
    y = y.cuda()

    ins_feat_out = torch.cat((ins_feat, x, x, x, y, y, y), 1)

    return ins_feat_out


class query_generator(nn.Module):
    def __init__(self, input, output, leaky=True):
        super(query_generator, self).__init__()
        self.proj1 = ConvBatchNormReLU(input+6, input+6, 3, 1, 1, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(input+6, input+6, 3, 1, 1, 1, leaky=leaky)
        self.proj3 = ConvBatchNormReLU(input+6, input+6, 3, 1, 1, 1, leaky=leaky)
        self.proj = nn.Conv2d(input+6, output, 1, 1, 0, 1)

    def forward(self, x):
        x = concat_coord(x)
        x = x + self.proj1(x)
        x = x + self.proj2(x)
        x = x + self.proj3(x)
        x = self.proj(x)
        return x


class KLM(nn.Module):
    def __init__(self, f_dim, feat_dim):
        super(KLM, self).__init__()
        self.lang_tf_enc = lang_tf_enc(f_dim, f_dim, f_dim, head_num=8)

        self.pos_embedding = PositionEmbeddingSine(f_dim)
        encoder_layer = TransformerEncoderLayer(f_dim, nhead=8, dim_feedforward=f_dim,
                                                dropout=0.1, activation='relu', normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=nn.LayerNorm(f_dim))

        self.fc_ker = nn.Linear(f_dim, feat_dim + feat_dim)
        self.fc_vis = nn.Linear(f_dim, feat_dim + feat_dim)
        self.ker_norm = nn.LayerNorm(feat_dim)
        self.vis_norm = nn.LayerNorm(feat_dim)

        self.channel_fc = nn.Linear(feat_dim, feat_dim)
        self.channel_norm = nn.LayerNorm(feat_dim)

        self.spatial_fc = nn.Linear(feat_dim, feat_dim)
        self.spatial_norm = nn.LayerNorm(feat_dim)

        self.out_fc = nn.Linear(feat_dim, f_dim)
        self.out_norm = nn.LayerNorm(f_dim)

        self.d_model = f_dim
        self.feat_dim = feat_dim
        self.resolution_size = 26

        #self.lang2kernel = nn.Linear(17, 16)

    def forward(self, kernel, lang_feat, visu_feat):
        # kernel    B x N x C
        # lang_feat B x T x C
        # visu_feat B x C x HW
        kernel = self.lang_tf_enc(kernel, lang_feat)   

        # B x N x C
        bs, c, hw = visu_feat.shape
        bq, nq, cq = kernel.shape
        bl, ll, cl = lang_feat.shape


        #lang = self.lang2kernel(lang_feat.permute(0,2,1))
        #lang = lang.permute(0,2,1)
        #kernel = lang

        # Image Attention
        visu_feat = visu_feat.permute(0, 2, 1)      
        # B x HW x C
        pos_embed = self.pos_embedding(visu_feat)   
        # B x HW x C

        visu_feat = visu_feat.transpose(0, 1)
        pos_embed = pos_embed.transpose(0, 1)               
        visu_feat_ = self.encoder(visu_feat, pos=pos_embed)  # HW x B x C
        visu_feat_ = visu_feat_.transpose(0, 1)     # B x HW x C

        #return kernel, visu_feat_.transpose(1, 2)
        
        
        # repeat visual feats
        visu_feat = visu_feat_.unsqueeze(dim=1) # B x 1 x HW x C
        kernel = kernel.unsqueeze(dim=2)        # B x N x  1 x C
        lang_feat = lang_feat.unsqueeze(dim=2)  # B x Q x  1 x C
        
        kernel_in = self.fc_ker(kernel)
        kernel_out = kernel_in[:, :, :, self.feat_dim:]
        kernel_in =  kernel_in[:, :, :, :self.feat_dim]

        vis_in = self.fc_vis(visu_feat)
        vis_out = vis_in[:, :, :, self.feat_dim:]
        vis_in  = vis_in[:, :, :, :self.feat_dim]

        gate_feat = self.ker_norm(kernel_in) * self.vis_norm(vis_in)
        #[B N HW 64]

        channel_gate = self.channel_norm(self.channel_fc(gate_feat))
        channel_gate = channel_gate.mean(2, keepdim=True)   
        channel_gate = torch.sigmoid(channel_gate)
        # B x N x 1 x C

        spatial_gate = self.spatial_norm(self.spatial_fc(gate_feat))
        # spatial_gate = spatial_gate.mean(3, keepdim=True)   
        spatial_gate = torch.sigmoid(spatial_gate)          
        # B x N x HW x C

        channel_gate = (1 + channel_gate) * kernel_out      # B x N x 1 x C
        channel_gate = channel_gate.squeeze(2)              # B x N x C

        spatial_gate = (1 + spatial_gate) * vis_out         # B x N x HW x C
        spatial_gate = spatial_gate.mean(2)                 # B x N x C
        
        gate_feat = (channel_gate + spatial_gate) / 2
        # [B N 64]
        gate_feat = self.out_fc(gate_feat)
        gate_feat = self.out_norm(gate_feat)
        gate_feat = F.relu(gate_feat)
        #[B N C]

        #visu_feat_.transpose(1, 2) [B C HW]
        return gate_feat, visu_feat_.transpose(1, 2)

class mask_decoder(nn.Module):
    def __init__(self, input_1, seg_out_stride=2, leaky=True):
        super(mask_decoder, self).__init__()
        self.proj1 = ConvBatchNormReLU(input_1, input_1 // 2, 3, 1, 1, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(input_1 // 2, input_1 // 2, 3, 1, 1, 1, leaky=leaky)
        
        self.proj3 = ConvBatchNormReLU(input_1 // 2, input_1 // 2, 3, 1, 1, 1, leaky=leaky)
        self.proj4 = ConvBatchNormReLU(input_1 // 2, input_1 // 2, 3, 1, 1, 1, leaky=leaky)
        self.proj5 = ConvBatchNormReLU(input_1 // 2, input_1 // 2 , 3, 1, 1, 1, leaky=leaky)
        

        self.proj = nn.Conv2d(input_1 // 2, 1, 3, 1, 1, 1)

    def forward(self, x, seg_out_stride):
        x = self.proj1(x)
        x = self.proj2(x)

        if seg_out_stride <= 8:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.proj3(x)

        if seg_out_stride <= 4:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.proj4(x)

        if seg_out_stride <= 2:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.proj5(x)
        

        x = self.proj(x)
        
        return x   

class KAM(nn.Module):
    def __init__(self, f_dim, num_query):
        super(KAM, self).__init__()

        self.k_size = 1

        self.proj = nn.Linear(26*26, f_dim)

        self.fc_k = nn.Linear(f_dim, f_dim)
        self.fc_m = nn.Linear(f_dim, f_dim)

        self.fc_fus = nn.Linear(f_dim * 2, f_dim)

        self.fc_out = nn.Linear(f_dim, 1)

        self.outproj = ConvBatchNormReLU(num_query, f_dim, 3, 1, 1, 1, leaky=True)
        self.maskproj = nn.Conv2d(f_dim, 1, 3, 1, 1, 1)

        self.bn = nn.BatchNorm2d(f_dim)

        self.mask_fcs = []
        for _ in range(3):
            self.mask_fcs.append(nn.Linear(f_dim, f_dim, bias=False))
            self.mask_fcs.append(nn.LayerNorm(f_dim))
            self.mask_fcs.append(nn.ReLU())
        self.mask_fcs = nn.Sequential(*self.mask_fcs)

        self.mask_decoder = mask_decoder(f_dim, seg_out_stride=2)


    def forward(self, kernel, visu_feat):
        # kernel [B N C]
        # visu_feat [B C HW]
        kernel = self.mask_fcs(kernel)

        B, N, C = kernel.shape
        kernel_ = kernel
        kernel = kernel.reshape(B, N, -1, C).permute(0, 1, 3, 2)    # B x N x C x 1
        kernel = kernel.reshape(B, N, C, self.k_size, self.k_size)  # B x N x C x 1 x 1
        #[B N C K K]
        visu_feat_ = visu_feat
        visu_feat = visu_feat.reshape(B, C, 26, 26)                 # B x C x H x W

        masks = []
        for i in range(B):
            masks.append(F.conv2d(visu_feat[i: i+1], kernel[i], padding=int(self.k_size // 2)))   # 1 x N x H x W
        masks = torch.cat(masks, dim=0)     # B x N x H x W

        b, n, h, w = masks.shape

        
        feats = masks.reshape(B, N, -1)     # B x N x HW
        feats = self.proj(feats)            # B x N x C

        weights_kern = F.relu(self.fc_k(kernel_))
        weights_mask = F.relu(self.fc_m(feats))

        weights = torch.cat([weights_kern, weights_mask], dim=-1)   # B x N x 2C

        weights = F.relu(self.fc_fus(weights))                      # B x N x C
        weights = self.fc_out(weights)                              # B x N x 1
        weights = F.softmax(weights, dim=1)                         # B x N x 1

        weights = weights.unsqueeze(-1)     # B x N x 1 x 1

        mask = weights * masks              # B x N x H x W
        mask = self.outproj(mask)           # B x C x H x W
        mask = self.maskproj(mask)          
        mask = F.sigmoid(mask)              # B x 1 x H x W

        visu_feat = visu_feat * mask        # B x C x H x W

        visu_feat = self.bn(visu_feat)
        visu_feat = visu_feat.reshape(B, C, -1) + visu_feat_
        visu_feat = F.relu(visu_feat)


        visu_feat = visu_feat.reshape(B, C, h, w)

        
        mask_out = self.mask_decoder(visu_feat, 2)

        return mask_out
    
        