import glob
import os
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline
from frag_utils import *
from torchvision.io import read_video, write_video
import random
import pdb

# suppress partial model loading warning
logging.set_verbosity_error()

VAE_BATCH_SIZE = 10

class FRAG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        
        ##################### Load stable diffusion ##############################
        sd_v = "stabilityai/stable-diffusion-2-1-base"
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_v, torch_dtype=torch.float16).to("cuda")
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(sd_v, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)

        ### Load video latent and total noise from (ddim latent - vae latent) ###
        self.ddim_latents_path = os.path.join(self.config["ddim_latents_path"], self.config["video_name"])
        video,_,meta = read_video(self.config["input_video"], output_format="TCHW")
        self.config['fps'] = meta['video_fps']
        frames = []
        for i in range(len(video)):
            image = T.ToPILImage()(video[i])
            image = image.resize((self.config["h"], self.config["w"]),  resample=Image.Resampling.LANCZOS)
            frame = image.convert('RGB')
            frames = frames + [frame]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        self.config['vid_len'] = len(video)
        self.latents = self.encode_imgs(frames, deterministic=True).to(torch.float16).to(self.device)
        self.eps = self.get_ddim_eps(self.latents).to(torch.float16).to(self.device)

        ################### Load source prompt and target prompt  ###############
        self.target_embeds = self.get_text_embeds(config["target_prompt"], config["negative_prompt"])
        src_prompt_path = os.path.join(self.ddim_latents_path, 'source_prompt', 'source_prompt.txt')
        with open(src_prompt_path, 'r') as f:
            src_prompt = f.read()
        self.config["source_prompt"] = src_prompt
        self.source_embeds = self.get_text_embeds(src_prompt, src_prompt).chunk(2)[0]

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=VAE_BATCH_SIZE, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, batch_size=VAE_BATCH_SIZE):
        latents = 1 / 0.18215 * latents
        imgs = []
        for i in range(0, len(latents), batch_size):
            imgs.append(self.vae.decode(latents[i:i + batch_size]).sample)
        imgs = torch.cat(imgs)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def get_ddim_eps(self, latent):
        noisest = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob(os.path.join(self.ddim_latents_path, 'latents', f'ddim_latents_*.pt'))])
        latents_path = os.path.join(self.ddim_latents_path, 'latents', f'ddim_latents_{noisest}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        alpha_prod_T = self.scheduler.alphas_cumprod[noisest]
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        eps = (noisy_latent - mu_T * latent) / sigma_T
        return eps

    @torch.no_grad()
    def denoise_step(self, x, t, indices):
        # register the time step and features in pnp injection modules
        latents_t_path = os.path.join(self.ddim_latents_path, 'latents', f'ddim_latents_{t}.pt')
        source_latents = torch.load(latents_t_path)[indices]
        z_ = source_latents
        latent_model_input = torch.cat([z_] + ([x] * 2))
        register_time(self, t.item())
        # compute text embeddings
        text_embed_input = torch.cat([self.source_embeds.repeat(len(indices), 1, 1),
                                      torch.repeat_interleave(self.target_embeds, len(indices), dim=0)])
        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']
        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)
        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def get_keyframe(self, batch_size):
        out = []
        for i in range(len(batch_size)):
            if i == 0:
                offset = 0
            else:
                offset = batch_size[i-1] + offset
            out.append(torch.randint(batch_size[i], (1,)).item() + offset)
        return torch.tensor(out)
    
    def DO_FRAG(self, x, x_prev, t):
        org_dtype = x.dtype
        x = FAR(x.to(dtype=torch.float32), x_prev.to(dtype=torch.float32), t)
        x = x.to(org_dtype)
        groups = TemporalGroup(x, t, self.config["min_size"], self.config["scheduler_beta"])
        return groups

    @torch.autocast(dtype=torch.float16, device_type='cuda')
    def batched_denoise_step(self, x, x_prev, t, indices):
        denoised_latents = []
        
        batch_size = self.DO_FRAG(x, x_prev, t)
        print("temporal group")
        print(batch_size)

        if self.config["module"] == 'propagation':
            key_idx = self.get_keyframe(batch_size)
            register_propagation(self, True)
            self.denoise_step(x[key_idx], t, indices[key_idx]) 
            register_propagation(self, False)
        
        for i in range(len(batch_size)):
            register_batch_idx(self, i)
            s = sum(batch_size[:i])
            e = s + batch_size[i]
            denoised_latents.append(self.denoise_step(x[s:e], t, indices[s:e]))

        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    def init_method(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_extended_attention_pnp(self, self.qk_injection_timesteps)
        register_conv_injection(self, self.conv_injection_timesteps)
        register_frag(self.unet, self.config["module"])

    def edit_video(self):
        # injection setting from prompt-to-prompt
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_method(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        # initial noise
        init_latents = self.scheduler.add_noise(self.latents, self.eps, self.scheduler.timesteps[0])
        # denoising
        indices = torch.arange(self.config["vid_len"])
        noise_latents = init_latents
        prev_latents = noise_latents
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
            noise_latents_next = self.batched_denoise_step(noise_latents, prev_latents, t, indices)
            prev_latents = noise_latents
            noise_latents = noise_latents_next
        denoised_latents = noise_latents
        # decoding
        edited_frames = self.decode_latents(denoised_latents)
        save_video(edited_frames, f'{self.config["output_path"]}/{self.config["video_name"]}.mp4', fps=self.config['fps'])
        with open(os.path.join(self.config["output_path"], "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
        print('Finish editing!')

def run(config):
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = FRAG(config)
    model.edit_video()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_sample3.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    video_name = Path(config["input_video"]).stem
    config["output_path"] = os.path.join(config["output_path"], video_name)
    config["video_name"] = video_name
    os.makedirs(config["output_path"], exist_ok=True)
    target_prompt = config["target_prompt"]
    print(f"Target prompt: {target_prompt}")
    run(config)
