import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from .sampler import Sampler
import pdb
from torch import nn
from utils.qam_utils import g_function_symbol,g_function_binary, general_m_qam_ber

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class BinaryDiffusionDSC(nn.Module):
    def __init__(self, H, denoise_fn, mask_id):
        super().__init__()

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.total_steps

        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = H.batch_size
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule

        self.loss_final = H.loss_final
        self.use_softmax = H.use_softmax

        self.p_flip = H.p_flip
        self.focal = H.focal
        self.aux = H.aux
        self.dataset = H.dataset
        self.guidance = H.guidance
        self.codebook_size = H.codebook_size
        self.block_size = H.block_size
        self.image_size = H.img_size
        self.qam_order = H.qam_order
  
        ######################################
        #gray code qam输入是nature二进制输出还是nature二进制，只不过中间变成symbole之前先变成了gray code，然后加噪声，然后解码的时候已经从gray code变回nature二进制了
        #所以我们是对用gray code qam编码加噪，再解码之后的引入噪声的自然二进制code进行diffusion去噪。由于BER也是用这个解码后的自然二进制code进行计算的
        #所以我们训练的时候这里已经是带噪点自然二进制的simulate了。
        #SNR_dB range
        self.Eb_N0_dB_range = torch.tensor([H.snr_range[0], H.snr_range[1]]) #2024/11/30 [0,15]

        self.Eb_N0_dB_values = torch.linspace(self.Eb_N0_dB_range[1], self.Eb_N0_dB_range[0],  self.num_timesteps)

        q_errot_t = general_m_qam_ber(self.Eb_N0_dB_values, self.qam_order)

        alpha_t = 1 - q_errot_t

        # 计算 gamma_t
        alpha_t_minus_1 = alpha_t[:-1]  # t-1 的 alpha_t
        alpha_t_current = alpha_t[1:]   # t 的 alpha_t

        # 根据公式计算 gamma_t
        gammas = (alpha_t_minus_1 - alpha_t_current) / (2 * alpha_t_minus_1 - 1)

        # 为了保持与原始 alpha_t 相同的长度，在开头添加一个零
        gammas = torch.cat((torch.tensor([0.0]), gammas))

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # Clear previously registered buffers
        self._buffers.clear()

        register_buffer('gammas', gammas)
        register_buffer('q_errot_t', q_errot_t)
        ######################################
    
    def get_q_error_t(self,t):
        return self.q_errot_t[t]
    

    def sample_time(self, b, device):
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return t

    def q_sample(self, x_0, t):  
        q_e_t = extract(self.q_errot_t, t, x_0.shape)
        x_t_in_prob = g_function_symbol(x_0, q_e_t)
        return x_t_in_prob

    def _train_loss(self, x_0, label=None, x_ct=None):
        x_0 = x_0 * 1.0
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t = self.sample_time(b, device)

        # make x noisy and denoise
        if x_ct is None:
            x_t = self.q_sample(x_0, t)

        x_t_in = torch.bernoulli(x_t)
        if label is not None:
            if self.guidance and np.random.random() < 0.1:
                label = None
            x_0_hat_logits = self._denoise_fn(idx=x_t_in, label=label, time_steps=t) 
        else:
            x_0_hat_logits = self._denoise_fn(x_t_in, time_steps=t)


        if self.p_flip:
            if self.focal >= 0:
                x_0_ = torch.logical_xor(x_0, x_t_in)*1.0
                kl_loss = focal_loss(x_0_hat_logits, x_0_, gamma=self.focal)
                x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
            else:
                x_0_hat_logits = x_t_in * ( - x_0_hat_logits) + (1 - x_t_in) * x_0_hat_logits
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')

        else:
            if self.focal >= 0:
                kl_loss = focal_loss(x_0_hat_logits, x_0, self.focal, gamma=self.focal)
            else:
                kl_loss = F.binary_cross_entropy_with_logits(x_0_hat_logits, x_0, reduction='none')

        if torch.isinf(kl_loss).max():
            pdb.set_trace()

        if self.loss_final == 'weighted':
            weight = (1 - (t / self.num_timesteps)).view(-1, 1, 1)
        elif self.loss_final == 'mean':
            weight = 1.0
        else:
            raise NotImplementedError
        
  
        loss = (weight * kl_loss).mean()
        kl_loss = kl_loss.mean()

        with torch.no_grad():
            if self.use_softmax:
                acc = (((x_0_hat_logits[..., 1] > x_0_hat_logits[..., 0]) * 1.0 == x_0.view(-1)) * 1.0).sum() / float(x_0.numel())
            else:
                acc = (((x_0_hat_logits > 0.0) * 1.0 == x_0) * 1.0).sum() / float(x_0.numel())

        if self.aux > 0:
     
            ftr = (((t-1)==0)*1.0).view(-1, 1, 1)

            x_0_l = torch.sigmoid(x_0_hat_logits)
            x_0_logits = torch.cat([x_0_l.unsqueeze(-1), (1-x_0_l).unsqueeze(-1)], dim=-1)

            x_t_logits = torch.cat([x_t_in.unsqueeze(-1), (1-x_t_in).unsqueeze(-1)], dim=-1)

            p_EV_qxtmin_x0 = self.q_sample(x_0_logits, t)


            gama_t = self.gammas[t]
            dim = x_t_logits.ndim - 1
            gama_t = gama_t.view(-1, *([1]*dim)) # TODO 替换成DSC的gama_t

            q_one_step = g_function_binary(x_t_logits, gama_t)

            unnormed_probs = p_EV_qxtmin_x0 * q_one_step
            unnormed_probs = unnormed_probs / (unnormed_probs.sum(-1, keepdims=True)+1e-6)
            unnormed_probs = unnormed_probs[...,0]
            
            x_tm1_logits = unnormed_probs * (1-ftr) + x_0_l * ftr
            x_0_gt = torch.cat([x_0.unsqueeze(-1), (1-x_0).unsqueeze(-1)], dim=-1)
            p_EV_qxtmin_x0_gt = self.q_sample(x_0_gt, t)
            unnormed_gt = p_EV_qxtmin_x0_gt * q_one_step
            unnormed_gt = unnormed_gt / (unnormed_gt.sum(-1, keepdims=True)+1e-6)
            unnormed_gt = unnormed_gt[...,0]

            x_tm1_gt = unnormed_gt

            if torch.isinf(x_tm1_logits).max() or torch.isnan(x_tm1_logits).max():
                pdb.set_trace()
                # 在计算二元交叉熵时禁用自动混合精度
            with torch.cuda.amp.autocast(enabled=False):
                aux_loss = F.binary_cross_entropy(x_tm1_logits.clamp(min=1e-6, max=(1.0-1e-6)), x_tm1_gt.clamp(min=0.0, max=1.0), reduction='none')

            aux_loss = (weight * aux_loss).mean()
            loss = self.aux * aux_loss + loss

        stats = {'loss': loss, 'bce_loss': kl_loss, 'acc': acc}

        if self.aux > 0:
            stats['aux loss'] = aux_loss
        return stats


    def sample(self, x_t = None, t_start=None, return_all=False, temp=1.0):
        device = 'cuda'

        sampling_steps = np.array(range(0, t_start))

        if return_all:
            x_all = [x_t]

        sampling_steps = sampling_steps[::-1]

        for i, t in enumerate(sampling_steps):
            t = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)

            x_0_logits = self._denoise_fn(x_t, time_steps=t)
            x_0_logits = x_0_logits / temp
                # scale by temperature

            x_0_logits = torch.sigmoid(x_0_logits)


            if self.p_flip:
                x_0_logits =  x_t * (1 - x_0_logits) + (1 - x_t) * x_0_logits

            if not t[0].item() == 0:
                t_p = torch.full((x_t.shape[0],), sampling_steps[i+1], device=device, dtype=torch.long)
                
                x_0_logits = torch.cat([x_0_logits.unsqueeze(-1), (1-x_0_logits).unsqueeze(-1)], dim=-1)
                x_t_logits = torch.cat([x_t.unsqueeze(-1), (1-x_t).unsqueeze(-1)], dim=-1)

                p_EV_qxtmin_x0 = self.q_sample(x_0_logits, t_p)
                q_one_step = x_t_logits
                
                gama_t = self.gammas[t]
                dim = x_t_logits.ndim - 1
                gama_t = gama_t.view(-1, *([1]*dim)) 
                q_one_step = g_function_symbol(q_one_step, gama_t) 

                unnormed_probs = p_EV_qxtmin_x0 * q_one_step
                unnormed_probs = unnormed_probs / unnormed_probs.sum(-1, keepdims=True)
                unnormed_probs = unnormed_probs[...,0]
                
                x_tm1_logits = unnormed_probs
                x_tm1_p = torch.bernoulli(x_tm1_logits)
            
            else:
                x_0_logits = x_0_logits
                x_tm1_p = (x_0_logits > 0.5) * 1.0

            x_t = x_tm1_p

            if return_all:
                x_all.append(x_t)
        if return_all:
            return torch.cat(x_all, 0)
        else:
            return x_t
        

    def forward(self, x, label=None, x_t=None):
        return self._train_loss(x, label, x_t)



def focal_loss(inputs, targets, alpha=-1, gamma=1):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    p_t = (1 - p_t)
    p_t = p_t.clamp(min=1e-6, max=(1-1e-6)) # numerical safety
    loss = ce_loss * (p_t ** gamma)
    if alpha == -1:
        neg_weight = targets.sum((-1, -2))
        neg_weight = neg_weight / targets[0].numel()
        neg_weight = neg_weight.view(-1, 1, 1)
        alpha_t = (1 - neg_weight) * targets + neg_weight * (1 - targets)
        loss = alpha_t * loss
    elif alpha > 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss

