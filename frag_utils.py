from typing import Type
import torch
import torch.fft as fft
import math
import os
import numpy as np
import copy
from torchvision.io import read_video, write_video
import pdb


def isinstance_str(x: object, cls_name: str):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False

def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity

def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }
    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)

# Adaptive Frequency Pass Filter (APF)
def Build_APF(shape, r, device, sigma=0.25, frame_wise=False):
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape, device=device)
    # frame-wise APF
    if frame_wise:
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    x = 2*h/H-1 # -1 < x < 1
                    y = 2*w/W-1 # -1 < y < 1
                    d = (x**2 + y**2)**(1/2)
                    if d < r[t]: 
                        mask[..., t,h,w] = 1
                    else:
                        mask[..., t,h,w] = math.exp(-1/(2*sigma**2) * (d-r[t])**2)
    else:
        r = r.mean()
        for h in range(H):
            for w in range(W):
                x = 2*h/H-1 # -1 < x < 1
                y = 2*w/W-1 # -1 < y < 1
                d = (x**2 + y**2)**(1/2)
                if d < r: 
                    mask[..., h,w] = 1
                else:
                    mask[..., h,w] = math.exp(-1/(2*sigma**2) * (d-r)**2)
    return mask

# Spatial Moment Adaption
def SMA(x, x_prev):
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    x_freq = torch.abs(x_freq)
    x_prev_freq = fft.fftn(x_prev, dim=(-2, -1))
    x_prev_freq = fft.fftshift(x_prev_freq, dim=(-2, -1))
    x_prev_freq = torch.abs(x_prev_freq)

    z_freq = x_freq - x_prev_freq
    z_freq = torch.mean(z_freq, dim=0)
    L,W,H = z_freq.shape
    z_quater = z_freq[:,int(W/2):,int(H/2):]
    y = torch.arange(int(H/2), device=x.device).view(1,-1)
    x = torch.arange(int(W/2), device=x.device).view(1,-1)
    z_quater_y = torch.sum(z_quater, dim=-1)
    z_quater_x = torch.sum(z_quater, dim=-2)
    
    z_sum_y = torch.sum(z_quater_y, dim=-1) + 0.00001
    zy_sum_y = torch.sum(z_quater_y*y, dim=-1)
    ry = torch.div(zy_sum_y, z_sum_y)
    ry = ry / int(H/2)
    
    z_sum_x = torch.sum(z_quater_x, dim=-1) + 0.00001
    zx_sum_x = torch.sum(z_quater_x*x, dim=-1)
    rx = torch.div(zx_sum_x, z_sum_x)
    rx = rx / int(W/2)

    r = torch.sqrt(rx*rx + ry*ry) + 0.2 # d_{0} = 0.2 standing for 0.2 * 32
    return r

# Frequency Adaptive Refinement
def FAR(x, x_prev, t):
    x = torch.transpose(x, 0, 1) # C, T, W, H
    x_prev = torch.transpose(x_prev, 0, 1)
    r = SMA(x, x_prev)
    APF = Build_APF(x.shape, r, x.device)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    # apply APF
    x_freq_apf = x_freq * APF
    # IFFT
    x_freq_apf = fft.ifftshift(x_freq_apf, dim=(-2, -1))
    y = fft.ifftn(x_freq_apf, dim=(-2, -1)).real
    y = torch.transpose(y, 0, 1)
    return y

def TemporalGroup(feat, t, min_g=4, sc_beta=1):
    mft = feat.mean(dim=-1).mean(dim=-1)
    L,D = mft.shape
    fl = mft[:-1]
    fr = mft[1:]
    dst_raw = torch.sqrt(torch.sum(torch.pow(torch.subtract(fl, fr), 2), dim=-1))
    
    # minimum size of group should be more than 2 (for propagation)
    rank0 = []
    # large F => fast but low qulaity
    F = min_g
    for i in range(0, L, F):
        #rank0.append([i,i+1])
        rank0.append([i+f for f in range(F)])
    if rank0[-1][-1] != L-1:
        #rank0[-1] = [L-1]
        a = rank0[-2][-1] + 1
        rank0[-1] = [i for i in range(a, L)]
    N = math.ceil(L/F)-1
    g_idx = torch.arange(N)*F + (F-1)
    dst = dst_raw[g_idx]

    ranked_linkage = torch.argsort(dst, descending=False)
    # scheduler
    n_root = len(dst)
    alpha = math.e - 1
    beta = sc_beta # 1<beta<3
    n_cut = math.ceil(n_root * math.log(alpha*t/(1000) + 1) * beta)

    # cluster
    merge_idx = ranked_linkage[:n_cut]
    group_idx = []
    for i in range(len(rank0)):
        group_idx.append([i])

    for i in range(len(merge_idx)):
        m_st = merge_idx[i]
        m_tgt = 0
        for j in range(len(group_idx)):
            if m_st in group_idx[j]:
                m_tgt = j
                break
        group_idx[m_tgt] = group_idx[m_tgt] + group_idx[m_tgt+1]
        group_idx.pop(m_tgt+1)
    
    rankn = []
    temp_group = []
    for i in range(len(group_idx)):
        tmp = []
        for j in range(len(group_idx[i])):
            tmp = tmp + rank0[group_idx[i][j]]
        rankn.append(tmp)
        temp_group.append(len(tmp))
    
    return temp_group

def register_propagation(diffusion_model, is_propagation):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "propagation_pass", is_propagation)
            
def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)

def register_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

def register_extended_attention_pnp(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                # inject unconditional
                q[n_frames:2 * n_frames] = q[:n_frames]
                k[n_frames:2 * n_frames] = k[:n_frames]
                # inject conditional
                q[2 * n_frames:] = q[:n_frames]
                k[2 * n_frames:] = k[:n_frames]
            k_source = k[:n_frames]
            k_uncond = k[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_cond = k[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            v_source = v[:n_frames]
            v_uncond = v[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_cond = v[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_source = self.head_to_batch_dim(q[:n_frames])
            q_uncond = self.head_to_batch_dim(q[n_frames:2 * n_frames])
            q_cond = self.head_to_batch_dim(q[2 * n_frames:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)


            q_src = q_source.view(n_frames, h, sequence_length, dim // h) # [4, 5, 4096, 64]
            k_src = k_source.view(n_frames, h, sequence_length, dim // h) # [4, 5, 4096, 64]
            v_src = v_source.view(n_frames, h, sequence_length, dim // h) # [4, 5, 4096, 64]

            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h) # [4, 5, 4096, 64]
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h) # [4, 5, 16384, 64]
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h) # [4, 5, 16384, 64]
            q_cond = q_cond.view(n_frames, h, sequence_length, dim // h) # [4, 5, 4096, 64]
            k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h) # [4, 5, 16384, 64]
            v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h) # [4, 5, 16384, 64]

            out_source_all = []
            out_uncond_all = []
            out_cond_all = []
            
            single_batch = n_frames<=12
            b = n_frames if single_batch else 1

            for frame in range(0, n_frames, b):
                out_source = []
                out_uncond = []
                out_cond = []
                for j in range(h):
                    sim_source_b = torch.bmm(q_src[frame: frame+ b, j], k_src[frame: frame+ b, j].transpose(-1, -2)) * self.scale # [4, 4096, 4096]
                    sim_uncond_b = torch.bmm(q_uncond[frame: frame+ b, j], k_uncond[frame: frame+ b, j].transpose(-1, -2)) * self.scale # [4, 4096, 16384]
                    sim_cond = torch.bmm(q_cond[frame: frame+ b, j], k_cond[frame: frame+ b, j].transpose(-1, -2)) * self.scale # [4, 4096, 16384]

                    out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[frame: frame+ b, j]))
                    out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[frame: frame+ b, j]))
                    out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[frame: frame+ b, j]))
                
                out_source = torch.cat(out_source, dim=0)
                out_uncond = torch.cat(out_uncond, dim=0) 
                out_cond = torch.cat(out_cond, dim=0) 
                if single_batch:
                    out_source = out_source.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_uncond = out_uncond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_cond = out_cond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                out_source_all.append(out_source)
                out_uncond_all.append(out_uncond)
                out_cond_all.append(out_cond)
            
            out_source = torch.cat(out_source_all, dim=0)
            out_uncond = torch.cat(out_uncond_all, dim=0)
            out_cond = torch.cat(out_cond_all, dim=0)
                
            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            # att1.forward = self attention on image patches
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def attention_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class temp_att(block_class):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
            low_freq=None,
        ) -> torch.Tensor:
            
            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 3
            hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            # change to TAV temporal attention
            attn_output = self.attn1(
                    norm_hidden_states.view(batch_size, sequence_length, dim),
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    **cross_attention_kwargs,
                )
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output
            
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                attn_output = attn_output.reshape(batch_size, sequence_length, dim)
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states
            return hidden_states
    return temp_att

def propagation_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class propagation(block_class):

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
            low_freq=None,
        ) -> torch.Tensor:
            
            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 3
            hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)

            if self.propagation_pass:
                self.keyframe_hidden_states = norm_hidden_states # [3, 4, 4096, 320]
            else:
                idx1 = []
                idx2 = [] 
                batch_idxs = [self.batch_idx]
                if self.batch_idx > 0:
                    batch_idxs.append(self.batch_idx - 1)
                sim = batch_cosine_sim(norm_hidden_states[0].reshape(-1, dim),
                                        self.keyframe_hidden_states[0][batch_idxs].reshape(-1, dim))
                if len(batch_idxs) == 2:
                    sim1, sim2 = sim.chunk(2, dim=1)
                    # sim: n_frames * seq_len, len(batch_idxs) * seq_len
                    idx1.append(sim1.argmax(dim=-1))  # n_frames * seq_len
                    idx2.append(sim2.argmax(dim=-1))  # n_frames * seq_len
                else:
                    idx1.append(sim.argmax(dim=-1))
                idx1 = torch.stack(idx1 * 3, dim=0) # 3, n_frames * seq_len
                idx1 = idx1.squeeze(1)
                if len(batch_idxs) == 2:
                    idx2 = torch.stack(idx2 * 3, dim=0) # 3, n_frames * seq_len
                    idx2 = idx2.squeeze(1)
            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.propagation_pass:
                # norm_hidden_states.shape = 3, n_frames * seq_len, dim
                self.attn_output = self.attn1(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        **cross_attention_kwargs,
                    )
                # 3, n_frames * seq_len, dim - > 3 * n_frames, seq_len, dim
                self.kf_attn_output = self.attn_output 
            else:
                batch_kf_size, _, _ = self.kf_attn_output.shape
                self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:,batch_idxs]
                # 3, n_frames, seq_len, dim --> 3, len(batch_idxs), seq_len, dim
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            if not self.propagation_pass:
                if len(batch_idxs) == 2:
                    attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                    attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))
                    s = torch.arange(0, n_frames).to(idx1.device) + batch_idxs[0] * n_frames
                    # distance from the keyframe
                    p1 = batch_idxs[0] * n_frames + n_frames // 2
                    p2 = batch_idxs[1] * n_frames + n_frames // 2
                    d1 = torch.abs(s - p1)
                    d2 = torch.abs(s - p2)
                    # weight
                    w1 = d2 / (d1 + d2)
                    w1 = torch.sigmoid(w1)
                    w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                    attn_output1 = attn_output1.view(3, n_frames, sequence_length, dim)
                    attn_output2 = attn_output2.view(3, n_frames, sequence_length, dim)
                    attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                else:
                    attn_output = self.attn_output[:,0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output = attn_output.view(3, n_frames, sequence_length, dim)
                attn_output = attn_output.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            else:
                attn_output = self.attn_output
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                attn_output = attn_output.reshape(batch_size, sequence_length, dim)
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = ff_output + hidden_states
            return hidden_states
    return propagation


def register_frag(model: torch.nn.Module, vqe_module):

    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            if vqe_module == 'propagation':
                vqe_block_fn = propagation_block 
            elif vqe_module == 'attention':
                vqe_block_fn = attention_block
            else:
                assert 0, 'No video quality enhancement module'

            module.__class__ = vqe_block_fn(module.__class__)

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model
