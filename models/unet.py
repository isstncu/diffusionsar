import math
import torch
import torch.nn as nn
from models.swin_transformer import BasicLayer
# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish SiLU()
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        # self.UP = nn.Sequential(
        #     nn.Conv2d(in_channels, 4*in_channels, 3, stride=1, padding=1),
        #     nn.PixelShuffle(2)
        # )
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # x = self.UP(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv: # default true
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        # self.layer = BasicLayer(dim=in_channels,
        #                         input_resolution=(resolution, resolution),
        #                         depth=2,
        #                         num_heads=in_channels // 32,
        #                         window_size=8,
        #                         mlp_ratio=1,
        #                         qkv_bias=True, qk_scale=None,
        #                         drop=0., attn_drop=0.,
        #                         drop_path=0.,
        #                         norm_layer=nn.LayerNorm)

    def forward(self, x):
        #print(x.shape)
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        #x_o = self.layer(x)

        return x+h_
        #return x_o


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult) # 128, 1, (1,1,2,3,4)
        num_res_blocks = config.model.num_res_blocks # 2
        attn_resolutions = config.model.attn_resolutions # [32,16,8]
        dropout = config.model.dropout # 0.0
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels # 1*2 = 2
        resolution = config.data.image_size # 128
        resamp_with_conv = config.model.resamp_with_conv # True

        self.ch = ch # 128
        self.temb_ch = self.ch*4 # 512
        self.num_resolutions = len(ch_mult) # 5
        self.num_res_blocks = num_res_blocks # 2
        self.resolution = resolution # 128
        self.in_channels = in_channels # 2

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])


        # downsampling,U-Net左部结构
        # ResBlock层(128,128)*2+DownSample层(128) +
        # ResBlock层(128,128)*2+DownSample层(128) +
        # ResBlock层(128,256)+Attention层(256)+ResBlock层(256,256)+Attention层(256)+DownSample层(256) +
        # ResBlock层(256,384)+Attention层(384)+ResBlock层(384,384)+Attention层(384)+DownSample层(384) +
        # ResBlock层(384,512)+Attention层(512)+ResBlock层(512,512)+Attention层(512)+
        self.conv_in = torch.nn.Conv2d(in_channels, # 2
                                       self.ch, # 128
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution # 128
        in_ch_mult = (1,)+ch_mult # (1, 1, 1, 2, 3, 4)
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions): # 5 (0,1,2,3,4)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level] # 128、128、128、256、384
            block_out = ch*ch_mult[i_level] # 128、128、256、384、512
            for i_block in range(self.num_res_blocks): # 2
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: # [32,16,8]
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1: # 4
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2 # 64、32、16、8
            self.down.append(down)

        # middle
        # 中间部分结构, U-net中间部分 ResBlock层(512,512)+
        #                         Attention层(512)+
        #                         ResBlock层(512,512)
        # 通道数和空间数都没有改变
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling,U-Net右部结构
        # ResBlock层(1024,512)+Attention层(512)+ResBlock层(1024,512)+Attention层(512)+ResBlock层(896,512)+Attention层(512)+UpSample层(512) +
        # ResBlock层(896,384)+Attention层(384)+ResBlock层(768,384)+Attention层(384)+ResBlock层(640,384)+Attention层(384)+UpSample层(384) +
        # ResBlock层(640,256)+Attention层(256)+ResBlock层(512,256)+Attention层(256)+ResBlock层(384,256)+Attention层(256)+UpSample层(256) +
        # ResBlock层(384,128)+ResBlock层(256,128)+ResBlock层(256,128)+UpSample层(128) +
        # ResBlock层(256,128)+ResBlock层(256,128)+ResBlock层(256,128)+
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): # 5 (4,3,2,1,0)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level] # 512,384,256,128,128
            skip_in = ch*ch_mult[i_level] # 512,384,256,128,128
            for i_block in range(self.num_res_blocks+1): # 3 (0,1,2)
                if i_block == self.num_res_blocks: # 2
                    skip_in = ch*in_ch_mult[i_level] # 384、256、128、128、128
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: # [8,16,32]
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2 # [16,32,64,128]
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)] # 先3×3卷积操作
        for i_level in range(self.num_resolutions): # 5 (0,1,2,3,4)
            for i_block in range(self.num_res_blocks): # 2 (0,1)
                h = self.down[i_level].block[i_block](hs[-1], temb) # Resblock
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h) # attention
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
