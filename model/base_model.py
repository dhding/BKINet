import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import *
from .modules import ConvBatchNormReLU, KLM, KAM, query_generator

import clip

class Simple_fusion(nn.Module):
    def __init__(self, visual_dim=1024, text_dim=768, proj_dim=1024, jemb_drop_out=0.1, leaky=True):
        super(Simple_fusion, self).__init__()
        self.proj_dim = proj_dim
        self.mapping_visu = ConvBatchNormReLU(visual_dim, proj_dim, 1, 1, 0, 1, leaky=leaky)
        self.lang_attn = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Dropout(jemb_drop_out),
            nn.Softmax(dim=1))
        
        self.lang_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.LeakyReLU(0.1))

        self.fusion = nn.Sequential(
            nn.BatchNorm2d(proj_dim),
            nn.LeakyReLU(0.1))
    
    def forward(self, visual_feat, lang_feat):
        # visual proj
        visual_feat_proj = self.mapping_visu(visual_feat) # [bt, 1024, 13, 13]

        lang_feat = lang_feat.squeeze(1)
        # lang proj
        lang_feat_new = self.lang_proj(lang_feat) #[bt, 1024]

        # fusion
        h, w = visual_feat.shape[-2], visual_feat.shape[-1]
        lang_feat_new_tile = lang_feat_new.view(-1, self.proj_dim, 1, 1).repeat(1, 1, h, w) # [bt, 1024, 13, 13]
        fusion_feat = lang_feat_new_tile * visual_feat_proj
        fusion_feat = self.fusion(fusion_feat)
        return fusion_feat

class up_proj_cat_proj(nn.Module):
    def __init__(self, input_1, input_2, do=512, leaky=True):
        super(up_proj_cat_proj, self).__init__()
        self.proj1 = ConvBatchNormReLU(input_2, input_2, 1, 1, 0, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(input_1+input_2, do, 1, 1, 0, 1, leaky=leaky)
    
    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        y = self.proj1(y)
        out = torch.cat([x,y], dim=1)
        out = self.proj2(out)
        return out

class pool_proj_cat_proj(nn.Module):
    def __init__(self, input_1, input_2, do=512, leaky=True):
        super(pool_proj_cat_proj, self).__init__()
        self.downsample = nn.AvgPool2d(2, 2)
        self.proj1 = ConvBatchNormReLU(input_2, do // 2, 1, 1, 0, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(do // 2, do, 3, 1, 1, 1, leaky=leaky)
        self.proj3 = ConvBatchNormReLU(input_1+do, do, 1, 1, 0, 1, leaky=leaky)

    def forward(self, x, y):
        y = self.downsample(y)
        y = self.proj1(y)
        y = self.proj2(y)
        output = self.proj3(torch.cat([x,y], dim=1))
        return output

class proj_cat(nn.Module):
    def __init__(self, input_1, input_2, do=512, leaky=True):
        super(proj_cat, self).__init__()
        self.proj1 = ConvBatchNormReLU(input_1, do // 2, 1, 1, 0, 1, leaky=leaky)
        self.proj2 = ConvBatchNormReLU(do // 2, do, 3, 1, 1, 1, leaky=leaky)

    def forward(self, x, y):
        x = self.proj1(x)
        x = self.proj2(x)
        output = torch.cat([x,y], dim=1)
        return output


class Model(nn.Module):
    def __init__(self, clip_model='RN50', tunelang=False, fusion_dim=2048, num_query=16, do=512, leaky=True, length=17):
        super(Model, self).__init__()

        self.tunelang = tunelang
        self.length = length

        ## Init Encoders
        clip_models = clip.load(clip_model, jit=False, device=torch.device("cpu"))[0].cuda()

        self.visumodel = clip_models.visual

        self.textmodel = clip_models.transformer
        self.textmodel_token_embedding = clip_models.token_embedding
        self.textmodel_pos_embed = clip_models.positional_embedding[:self.length, :].unsqueeze(0)
        self.textmodel_ln_final = clip_models.ln_final
        self.textdim = self.textmodel_pos_embed.shape[-1]
        for module in self.textmodel.resblocks:
            module.attn_mask = self.build_attention_mask()

        ## Fusion
        self.fusion = Simple_fusion(visual_dim=fusion_dim, text_dim=self.textdim, proj_dim=fusion_dim)
        self.up_proj_cat_proj_1 = up_proj_cat_proj(input_1=fusion_dim, input_2=fusion_dim // 2, do=fusion_dim // 2)
        self.pool_proj_cat_proj_2 = pool_proj_cat_proj(input_1=fusion_dim // 2, input_2=fusion_dim // 4, do=do)
        self.proj_cat = proj_cat(input_1=fusion_dim // 2, input_2=fusion_dim // 4, do=do)
        self.up_proj_cat_2 = up_proj_cat_proj(input_1=fusion_dim, input_2=fusion_dim // 2, do=do)     
        self.proj_0 = ConvBatchNormReLU(do, do, 1, 1, 0, 1, leaky=leaky)
        
        ## Main
        self.query_generator = query_generator(input=do, output=num_query)

        ## Align dim
        f_dim = 512
        self.fc_1 = nn.Linear(26*26, f_dim, bias=False)
        self.fc_2 = nn.Linear(f_dim, f_dim, bias=False)
        self.norm1 = nn.LayerNorm(f_dim)
        self.norm2 = nn.LayerNorm(f_dim)

        self.KLM = KLM(f_dim, 64)
        self.KAM = KAM(f_dim, 16)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.length, self.length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return 

    def forward(self, image, word_id, word_mask):
        ## Visual Module
        batch_size = image.size(0)

        ## Extract features from vision
        raw_fvisu = []

        x = self.visumodel.relu1(self.visumodel.bn1(self.visumodel.conv1(image)))
        x = self.visumodel.relu2(self.visumodel.bn2(self.visumodel.conv2(x)))
        x = self.visumodel.relu3(self.visumodel.bn3(self.visumodel.conv3(x)))
        x = self.visumodel.avgpool(x)

        x = self.visumodel.layer1(x)
        x = self.visumodel.layer2(x)
        raw_fvisu = [x] + raw_fvisu
        x = self.visumodel.layer3(x)
        raw_fvisu = [x] + raw_fvisu
        x = self.visumodel.layer4(x)
        raw_fvisu = [x] + raw_fvisu
        
        # Extract features from lang
        raw_fword = self.textmodel_token_embedding(word_id).squeeze(1)
        raw_fword = raw_fword + self.textmodel_pos_embed
        raw_fword = raw_fword.permute(1, 0, 2)
        raw_fword = self.textmodel(raw_fword)
        raw_fword = raw_fword.permute(1, 0, 2)

        raw_fword = self.textmodel_ln_final(raw_fword)
        
        if not self.tunelang:
            raw_fword = raw_fword.detach()

        eos_token = raw_fword[:,-1:,:]
        
        F_s = self.fusion(raw_fvisu[0], eos_token)
        F_m = self.up_proj_cat_proj_1(F_s, raw_fvisu[1])
        F_l = self.pool_proj_cat_proj_2(F_m, raw_fvisu[2])
        #[B C H W]
        Fm_mid = self.proj_cat(F_m, F_l)
        Fm_top = self.up_proj_cat_2(F_s, Fm_mid)
        F_tf = self.proj_0(Fm_top) # F_I

        # Generate query from language
        kernel = self.query_generator(F_l)
        #[B N H W]

        # Main body
        b,  c,  h,  w = F_tf.shape
        bq, cq, hq ,wq = kernel.shape

        flatten_length = h*w
        visu_feat = F_tf.reshape(b, c, flatten_length)

        # Proposed Modules
        kernel = kernel.reshape(bq, cq, flatten_length) # B x Q x HW
        kernel = F.relu(self.fc_1(kernel))              # B x Q x C

        lang_feat = F.relu(self.fc_2(raw_fword))        # B x L x C

        
        #kernel [B N C]
        #lang_feat [B T C]
        #visu_feat [B C HW]
        kernel, visu_feat = self.KLM(kernel, lang_feat, visu_feat)

        mask_out = self.KAM(kernel, visu_feat)

        return mask_out
