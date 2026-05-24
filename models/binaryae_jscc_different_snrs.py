'''
Binary Autoencoder, adapted from the original created by the Taming Transformers authors:
https://github.com/CompVis/taming-transformers/blob/master/taming/models/vqgan.py

'''

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffaug import DiffAugment
from utils.vqgan_utils import normalize, swish, adopt_weight, hinge_d_loss, calculate_adaptive_weight
from utils.log_utils import log
import torch.distributed as dist
import math
from utils.qam_utils import Eb_No_dB_to_sigma_in_M_QAM, qam_constellation, mapping, demapping, SNR_dB_to_sigma_in_M_QAM

def get_image_from_binary_code(generator,  binary_code = None, embedding_weight = None):  # 4月七号测试实值DDPM，即不走输入01code那一套了
    generator.eval()
   
    binary_latent_code = binary_code
    print("latents shape:")
    print(binary_latent_code.shape)
    latent_shape = [1,16,16]
    use_tanh = False
    norm_first = True
    codebook_size = 64
    with torch.cuda.amp.autocast():
        size = min(25, binary_latent_code.shape[0])
        if latent_shape[-1] == 32:
            size = 5
        images = []
        for i in range(len(binary_latent_code)//size):
            latent = binary_latent_code[i*size : (i+1)*size]

            latent = (latent * 1.0) 

            if use_tanh:
                latent = (latent - 0.5) * 2.0
            # latents = latents / (latents.sum(dim=-1, keepdim=True)+1e-6)
            if not norm_first:
                latent = latent / float(codebook_size)
            latent = latent @ embedding_weight

            latent = latent.permute(0,2,1)
            latent = latent.reshape(*latent.shape[:-1], latent_shape[1], latent_shape[2])
            img = generator(latent.float())
            images.append(img)
        images = torch.cat(images, 0)

    return images


class BinaryQuantizerWithoutSampler(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, H, use_tanh=False):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.H = H
        act = nn.Sigmoid
        if use_tanh:
            act = nn.Tanh
        self.proj = nn.Sequential(
            nn.Conv2d(num_hiddens, codebook_size, 1),  # projects last encoder layer to quantized logits
            act(),
            )
        self.embed = nn.Embedding(codebook_size, emb_dim)
        self.use_tanh = use_tanh

    def quantizer(self, x, deterministic=False):
        if self.use_tanh:
            x = x * 0.5 + 0.5
            if deterministic:
                x = (x > 0.5) * 1.0 
            else:
                x = torch.bernoulli(x)
            x = (x - 0.5) * 2.0
            return x
            
        else:
            if deterministic:
                x = (x > 0.5) * 1.0 
                return x
            else:
                return torch.bernoulli(x)

    def forward(self, h, deterministic=False):

        z = self.proj(h)

        # code_book_loss = F.binary_cross_entropy_with_logits(z, (torch.sigmoid(z.detach())>0.5)*1.0)
        code_book_loss = (torch.sigmoid(z) * (1 - torch.sigmoid(z))).mean()

        z_b = self.quantizer(z, deterministic=deterministic)

        ##### jscc 给encode出来的binary code加一个fixed的snr的噪声#######
        code = z_b
        device = code.device
        #后面两个维度合并成一个维度
        shape = code.shape
   
        # # make x noisy and denoise 
        # sigma_t = Eb_No_dB_to_sigma_in_M_QAM(torch.tensor(self.H.snr), self.H.qam_order).to(device)

        # code_flatten = code.flatten()

        # constellation = qam_constellation(self.H.qam_order).to(device)
    
        # # Map binary data to QAM  constellation symbols
        # symbols = mapping(code_flatten, constellation, self.H.qam_order)

        #     # Simulate AWGN channel
        # noise = sigma_t * (torch.randn_like(symbols) + 1j * torch.randn_like(symbols))
        
        # received_symbols = symbols + noise

        # # turn noisy constellation symbols into binary code
        # noise_binary = demapping(received_symbols, constellation, self.H.qam_order)

        # ... 在主流程中添加噪声的部分修改如下 ...

        sigma_t = SNR_dB_to_sigma_in_M_QAM(self.H.snr, self.H.qam_order).to(device)

        # 确保 symbols 展平
        code_flatten = code.flatten()
        constellation = qam_constellation(self.H.qam_order).to(device)

        # Map binary data to QAM constellation symbols (复数)
        symbols = mapping(code_flatten, constellation, self.H.qam_order)

        # 修复 PyTorch 复数 randn_like 的 Bug
        # 显式生成与 symbols shape 相同的实数分布
        noise_real = torch.randn(symbols.shape, dtype=torch.float32, device=device)
        noise_imag = torch.randn(symbols.shape, dtype=torch.float32, device=device)

        # 组合成复数噪声
        noise = sigma_t * (noise_real + 1j * noise_imag)

        received_symbols = symbols + noise

        # 后面保持不变
        noise_binary = demapping(received_symbols, constellation, self.H.qam_order)

        noise_binary = noise_binary.view(shape)

        z_b = noise_binary
        

        ##################################################

        z_flow = z_b.detach() + z - z.detach() 

        z_q = torch.einsum("b n h w, n d -> b d h w", z_flow, self.embed.weight)
        return z_q, code_book_loss, {
            "binary_code": z_b.detach()
        }, z_b.detach()



class BinaryQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, use_tanh=False):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        act = nn.Sigmoid
        if use_tanh:
            act = nn.Tanh
        self.proj = nn.Sequential(
            nn.Conv2d(num_hiddens, codebook_size, 1),  # projects last encoder layer to quantized logits
            act(),
            )
        self.embed = nn.Embedding(codebook_size, emb_dim)
        self.use_tanh = use_tanh

    def quantizer(self, x, deterministic=False):
        if self.use_tanh:
            x = x * 0.5 + 0.5
            if deterministic:
                x = (x > 0.5) * 1.0 
            else:
                x = torch.bernoulli(x)
            x = (x - 0.5) * 2.0
            return x
            
        else:
            if deterministic:
                x = (x > 0.5) * 1.0 
                return x
            else:
                return torch.bernoulli(x)

    def forward(self, h, deterministic=False):

        z = self.proj(h)

        # code_book_loss = F.binary_cross_entropy_with_logits(z, (torch.sigmoid(z.detach())>0.5)*1.0)
        code_book_loss = (torch.sigmoid(z) * (1 - torch.sigmoid(z))).mean()

        z_b = self.quantizer(z, deterministic=deterministic)

        z_flow = z_b.detach() + z - z.detach()

        z_q = torch.einsum("b n h w, n d -> b d h w", z_flow, self.embed.weight)

        return z_q,  code_book_loss, {
            "binary_code": z_b.detach()
        }, z_b.detach()


class BinaryQuantizerBPSK(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, use_tanh=False):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        act = nn.Sigmoid
        if use_tanh:
            act = nn.Tanh
        self.proj = nn.Sequential(
            nn.Conv2d(num_hiddens, codebook_size, 1),  # projects last encoder layer to quantized logits
            act(),
            )
        self.embed = nn.Embedding(codebook_size, emb_dim)
        self.use_tanh = use_tanh
        self.alpha=150.0
        self.gamma=3.0

    def quantizer(self, x, deterministic=False):
        if self.use_tanh:
            x = x * 0.5 + 0.5
            if deterministic:
                x = (x > 0.5) * 1.0 
            else:
                x = torch.bernoulli(x)
            x = (x - 0.5) * 2.0
            return x
            
        else:
            if deterministic:
                x = (x > 0.5) * 1.0 
                return x
            else:
                return torch.bernoulli(x)

    def forward(self, h, deterministic=False,sampler = None, generator = None, mode = None):

        z = self.proj(h)

        # code_book_loss = F.binary_cross_entropy_with_logits(z, (torch.sigmoid(z.detach())>0.5)*1.0)
        code_book_loss = (torch.sigmoid(z) * (1 - torch.sigmoid(z))).mean()

        z_b = self.quantizer(z, deterministic=deterministic)

        #add noise -> denoise bpsk
        if (sampler is not None) and (mode == "training"):
            b, device = z_b.size(0), z_b.device
            # choose what time steps to compute loss at
            #TODO z_b最后两个维度合并成一个维度

            #后面两个维度合并成一个维度
            z_b = z_b.view(z_b.size(0), z_b.size(1), -1)
            #最后两个维度换一下顺序
            z_b = z_b.permute(0, 2, 1)
            

            t = sampler.sample_time_joint_train(device)

            z_b_noise = sampler.bpsk_noise_sheduler_x0_xt(z_b, t)
            noise_latent_t = [z_b_noise,t,z_b]
         
            z_b_denoise = sampler.p_sample_loop_train(noise_latent_t = noise_latent_t)



            ###################### 新增 diffusion loss 2024/6/14
            # TODO z_b 和z_b_denoise 都是元素是0,1的tensor，现在希望计算他们之间的loss

             # Cloning tensors for autograd
            # z_b_denoise_clone = z_b_denoise.clone().float()
            # z_b_clone = z_b.clone().float()
            
            # Computing the loss
#             diffusion_kl_loss = F.binary_cross_entropy_with_logits(z_b_denoise_clone, z_b_clone, reduction='none')
        
#             diffusion_kl_loss = diffusion_kl_loss.mean()
            ######################
            
    
            # #存一张图下来看看 待会删掉
            from utils.log_utils import save_results_modified
            removed_noise_image = get_image_from_binary_code(generator, binary_code = z_b_denoise, embedding_weight = self.embed.weight)
            save_results_modified(removed_noise_image, f'steps_{t}', 8500, "experiments/train_logs/jointly_train_decode_only_snr_-10", True,0,0)
  

            
            z_b_denoise = z_b_denoise.permute(0, 2, 1)
            last_two_dim = int(math.sqrt(z_b_denoise.size(2)))
            z_b_denoise = z_b_denoise.view(z_b_denoise.size(0), z_b_denoise.size(1), last_two_dim, last_two_dim)
            
            #训bpsk代码
            z_flow = z_b_denoise.detach() + z - z.detach()
        else:
             #原代码，保留在这，后面要是采样还要恢复回去,联合BLD训练的时候不用它
            z_flow = z_b.detach() + z - z.detach() 

        z_q = torch.einsum("b n h w, n d -> b d h w", z_flow, self.embed.weight)
        return z_q, code_book_loss, {
            "binary_code": z_b.detach()
        }, z_b.detach()


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        # x = swish(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        # x = swish(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
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
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, out_channels, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.nf = H.nf
        self.ch_mult = H.ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = int(H.res_blocks * H.gen_mul)
        self.resolution = H.img_size
        self.attn_resolutions = H.attn_resolutions
        self.in_channels = H.emb_dim
        self.out_channels = H.n_channels
        self.norm_first = H.norm_first
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        blocks = []
        # initial conv
        if self.norm_first:
            blocks.append(normalize(self.in_channels))
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

        # used for calculating ELBO - fine tuned after training
        self.logsigma = nn.Sequential(
                            nn.Conv2d(block_in_ch, block_in_ch, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(block_in_ch, H.n_channels, kernel_size=1, stride=1, padding=0)
                        ).cuda()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class BinaryAutoEncoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.in_channels = H.n_channels
        self.nf = H.nf
        self.n_blocks = H.res_blocks
        self.codebook_size = H.codebook_size
        self.embed_dim = H.emb_dim
        self.ch_mult = H.ch_mult
        self.resolution = H.img_size
        self.attn_resolutions = H.attn_resolutions
        self.quantizer_type = H.quantizer
        self.beta = H.beta
        self.gumbel_num_hiddens = H.emb_dim
        self.deterministic = H.deterministic
        self.snr = H.snr
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )
        self.generator = Generator(H)
        self.quantize = BinaryQuantizerWithoutSampler(self.codebook_size, self.embed_dim, self.embed_dim, use_tanh=H.use_tanh, H=H)
        self.mode = H.mode
    

        # print(self.encoder)
        # print(self.quantize)
        # print(self.generator)

    def forward(self, x, code_only=False, code=None):
        if code is None:
            x = self.encoder(x)
            quant, codebook_loss, quant_stats, binary = self.quantize(x, deterministic=self.deterministic)
            if code_only:
                return binary
        else:
            #用 0 1的code去选实数的code  [Batch,256，64] * [Batch,64,256] = [Batch,256,256]  后面会再转化为-> [Batch,256,16 ,16] 
            quant = torch.einsum("b n h w, n d -> b d h w", code, self.quantize.embed.weight)
            codebook_loss, quant_stats = None, None
        x = self.generator(quant)
        return x, codebook_loss, quant_stats


# patch based discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf, n_layers=3):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BinaryGAN(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.ae = BinaryAutoEncoder(H)
        self.disc = Discriminator(
            H.n_channels,
            H.ndf,
            n_layers=H.disc_layers
        )

        if dist.get_rank() == 0:
            self.perceptual = lpips.LPIPS(net="vgg")
        
        dist.barrier()
        
        self.perceptual = lpips.LPIPS(net="vgg")
        self.perceptual_weight = H.perceptual_weight
        self.disc_start_step = H.disc_start_step
        self.disc_weight_max = H.disc_weight_max
        self.diff_aug = H.diff_aug
        self.policy = "color,translation"

        self.code_weight = H.code_weight
    

    def forward(self, x, step):
        return self.train_iter(x, step)
        
    def train_iter(self, x, step):
        stats = {}

        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon/perceptual loss
        # recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        recon_loss = nn.SmoothL1Loss()(x.contiguous(), x_hat.contiguous())

        p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
        nll_loss = recon_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        # augment for input to discriminator
        if self.diff_aug:
            x_hat_pre_aug = x_hat.detach().clone()
            x_hat = DiffAugment(x_hat, policy=self.policy)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.ae.generator.blocks[-1].weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer, self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
        loss = nll_loss + d_weight * g_loss + self.code_weight * codebook_loss

        stats["loss"] = loss
        stats["l1"] = recon_loss.mean().item()
        stats["perceptual"] = p_loss.mean().item()
        stats["nll_loss"] = nll_loss.item()
        stats["g_loss"] = g_loss.item()
        stats["d_weight"] = d_weight
        stats["codebook_loss"] = codebook_loss.item()
        stats["latent_ids"] = quant_stats["binary_code"]

        if "mean_distance" in stats:
            stats["mean_code_distance"] = quant_stats["mean_distance"].item()
        

        if self.diff_aug:
            x_hat = x_hat_pre_aug

        return x_hat, stats

    def disc_iter(self, x_hat, x, states):
        if self.diff_aug:
            logits_real = self.disc(DiffAugment(x.contiguous().detach(), policy=self.policy))
        else:
            logits_real = self.disc(x.contiguous().detach())
        logits_fake = self.disc(x_hat.contiguous().detach())  # detach so that generator isn"t also updated
        d_loss = hinge_d_loss(logits_real, logits_fake)
        states["d_loss"] = d_loss
        return states

    @torch.no_grad()
    def val_iter(self, x, step):
        stats = {}
        # update gumbel softmax temperature based on step. Anneal from 1 to 1/16 over 150000 steps
        if self.ae.quantizer_type == "gumbel":
            self.ae.quantize.temperature = max(1/16, ((-1/160000) * step) + 1)
            stats["gumbel_temp"] = self.ae.quantize.temperature

        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon/perceptual loss
        # recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        recon_loss = nn.SmoothL1Loss()(x.contiguous(), x_hat.contiguous())

        p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
        nll_loss = recon_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)

        stats["l1"] = recon_loss.mean().item()
        stats["perceptual"] = p_loss.mean().item()
        stats["nll_loss"] = nll_loss.item()
        stats["g_loss"] = g_loss.item()
        stats["codebook_loss"] = codebook_loss.item()
        stats["latent_ids"] = quant_stats["binary_code"]

        return x_hat, stats
    
#ae with sampler
class BinaryAutoEncoderBPSK(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.in_channels = H.n_channels
        self.nf = H.nf
        self.n_blocks = H.res_blocks
        self.codebook_size = H.codebook_size
        self.embed_dim = H.emb_dim
        self.ch_mult = H.ch_mult
        self.resolution = H.img_size
        self.attn_resolutions = H.attn_resolutions
        self.quantizer_type = H.quantizer
        self.beta = H.beta
        self.gumbel_num_hiddens = H.emb_dim
        self.deterministic = H.deterministic
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )

        self.generator = Generator(H)
        self.quantize = BinaryQuantizerBPSK(self.codebook_size, self.embed_dim, self.embed_dim, use_tanh=H.use_tanh)
        self.mode = H.mode
    
        from hparams import get_sampler_hparams_bpsk
        from utils.sampler_utils import get_sampler
        from utils.log_utils import load_model_directly

        sampler_H = get_sampler_hparams_bpsk()
        self.sampler =  get_sampler(sampler_H, self.quantize.embed.weight).cuda()
        if H.load_sampler_step > 0:
            device = "cuda:0"
            allow_mismatch = H.allow_mismatch
            if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
                try:
                    self.sampler = load_model_directly(
                         self.sampler, f'{H.sampler}_ema', H.load_sampler_step, H.load_sampler_dir, device=device, allow_mismatch=allow_mismatch)
                    print("sampler_created!")
                except Exception:
                    print("No EMA model found, starting EMA from model load point")
        
                 
        # print(self.encoder)
        # print(self.quantize)
        # print(self.generator)

    def forward(self, x, code_only=False, code=None):
        if code is None:
            x = self.encoder(x)
            quant, codebook_loss, quant_stats, binary = self.quantize(x, deterministic=self.deterministic, sampler = self.sampler
                                                                      , generator = self.generator, mode = self.mode)
            if code_only:
                return binary
        else:
            #用 0 1的code去选实数的code  [Batch,256，64] * [Batch,64,256] = [Batch,256,256]  后面会再转化为-> [Batch,256,16 ,16] 
            quant = torch.einsum("b n h w, n d -> b d h w", code, self.quantize.embed.weight)
            codebook_loss, quant_stats = None, None
        x = self.generator(quant)
        return x, codebook_loss, quant_stats


class BinaryGANBPSK(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.ae = BinaryAutoEncoderBPSK(H)
        self.disc = Discriminator(
            H.n_channels,
            H.ndf,
            n_layers=H.disc_layers
        )

        if dist.get_rank() == 0:
            self.perceptual = lpips.LPIPS(net="vgg")
        
        dist.barrier()
        
        self.perceptual = lpips.LPIPS(net="vgg")
        self.perceptual_weight = H.perceptual_weight
        self.disc_start_step = H.disc_start_step
        self.disc_weight_max = H.disc_weight_max
        self.diff_aug = H.diff_aug
        self.policy = "color,translation"

        self.code_weight = H.code_weight
    

    def forward(self, x, step):
        return self.train_iter(x, step)
        
    def train_iter(self, x, step):
        stats = {}

        ########################  新增联合训练的diffusion_kl_loss 2024/6/14
        
        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon/perceptual loss
        # recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        recon_loss = nn.SmoothL1Loss()(x.contiguous(), x_hat.contiguous())
        p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
        nll_loss = recon_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        # augment for input to discriminator
        if self.diff_aug:
            x_hat_pre_aug = x_hat.detach().clone()
            x_hat = DiffAugment(x_hat, policy=self.policy)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)
        last_layer = self.ae.generator.blocks[-1].weight
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer, self.disc_weight_max)
        d_weight *= adopt_weight(1, step, self.disc_start_step)
#         loss = nll_loss + d_weight * g_loss + self.code_weight * codebook_loss
        loss = nll_loss + d_weight * g_loss 

        stats["loss"] = loss
        stats["l1"] = recon_loss.mean().item()
        stats["perceptual"] = p_loss.mean().item()
        stats["nll_loss"] = nll_loss.item()
        stats["g_loss"] = g_loss.item()
        stats["d_weight"] = d_weight
        stats["codebook_loss"] = codebook_loss.item()
        stats["latent_ids"] = quant_stats["binary_code"]
        # stats["diffusion_kl_loos"] = diffusion_kl_loss

        if "mean_distance" in stats:
            stats["mean_code_distance"] = quant_stats["mean_distance"].item()
        

        if self.diff_aug:
            x_hat = x_hat_pre_aug

        return x_hat, stats

    def disc_iter(self, x_hat, x, states):
        if self.diff_aug:
            logits_real = self.disc(DiffAugment(x.contiguous().detach(), policy=self.policy))
        else:
            logits_real = self.disc(x.contiguous().detach())
        logits_fake = self.disc(x_hat.contiguous().detach())  # detach so that generator isn"t also updated
        d_loss = hinge_d_loss(logits_real, logits_fake)
        states["d_loss"] = d_loss
        return states

    @torch.no_grad()
    def val_iter(self, x, step):
        stats = {}
        # update gumbel softmax temperature based on step. Anneal from 1 to 1/16 over 150000 steps
        if self.ae.quantizer_type == "gumbel":
            self.ae.quantize.temperature = max(1/16, ((-1/160000) * step) + 1)
            stats["gumbel_temp"] = self.ae.quantize.temperature

        x_hat, codebook_loss, quant_stats = self.ae(x)

        # get recon/perceptual loss
        recon_loss = torch.abs(x.contiguous() - x_hat.contiguous())  # L1 loss
        p_loss = self.perceptual(x.contiguous(), x_hat.contiguous())
        nll_loss = recon_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        # update generator
        logits_fake = self.disc(x_hat)
        g_loss = -torch.mean(logits_fake)

        stats["l1"] = recon_loss.mean().item()
        stats["perceptual"] = p_loss.mean().item()
        stats["nll_loss"] = nll_loss.item()
        stats["g_loss"] = g_loss.item()
        stats["codebook_loss"] = codebook_loss.item()
        stats["latent_ids"] = quant_stats["binary_code"]

        return x_hat, stats


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.99, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability is 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
