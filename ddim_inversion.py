from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
logging.set_verbosity_error()

import os
import shutil
import numpy as np
import random
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from torchvision.io import read_video, write_video
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import yaml
import pdb

@torch.no_grad()
def encode_imgs(vae, imgs, batch_size=10, deterministic=True):
    imgs = 2 * imgs - 1
    latents = []
    for i in range(0, len(imgs), batch_size):
        posterior = vae.encode(imgs[i:i + batch_size]).latent_dist
        latent = posterior.mean if deterministic else posterior.sample()
        latents.append(latent * 0.18215)
    latents = torch.cat(latents)
    return latents

@torch.no_grad()
def get_text_embeds(tokenizer, text_encoder, prompt, negative_prompt, device):
    text_input = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    uncond_input = tokenizer(negative_prompt, padding='max_length', max_length=tokenizer.model_max_length,
                                  return_tensors='pt')
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings

@torch.no_grad()
def ddim_inversion(cond, latent, scheduler, unet, save_path, save_steps, batch_size=100):
    timesteps = reversed(scheduler.timesteps)
    for i, t in enumerate(tqdm(timesteps)):
        for b in range(0, latent.shape[0], batch_size):
            latent_batch = latent[b:b + batch_size]
            model_input = latent_batch
            cond_batch = cond.repeat(latent_batch.shape[0], 1, 1)
                                                                
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = unet(model_input, t, encoder_hidden_states=cond_batch).sample
            pred_x0 = (latent_batch - sigma_prev * eps) / mu_prev
            latent[b:b + batch_size] = mu * pred_x0 + sigma * eps
        if t in save_steps:
            torch.save(latent, os.path.join(save_path, f'ddim_latents_{t}.pt'))
    torch.save(latent, os.path.join(save_path, f'ddim_latents_{t}.pt'))

def run(opt):
    # seed setting
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # out dir setting
    save_path = os.path.join(opt.out_dir, opt.video_name, 'latents')
    save_prompt_path = os.path.join(opt.out_dir, opt.video_name, 'source_prompt')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_prompt_path, exist_ok=True)

    with open(os.path.join(save_prompt_path, 'source_prompt.txt'), 'w') as f:
        f.write(opt.source_prompt)

    ######### Stable Diffusion Setting ###################
    # sd version, empirically 2.1 gives more better results
    sd_v = "stabilityai/stable-diffusion-2-1-base"
    ddim_scheduler = DDIMScheduler.from_pretrained(sd_v, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(sd_v, subfolder="vae", revision="fp16", torch_dtype=torch.float16).to(opt.device)
    tokenizer = CLIPTokenizer.from_pretrained(sd_v, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_v, subfolder="text_encoder", revision="fp16", torch_dtype=torch.float16).to(opt.device)
    unet = UNet2DConditionModel.from_pretrained(sd_v, subfolder="unet", revision="fp16", torch_dtype=torch.float16).to(opt.device)
    
    ######### input video latent #########################
    video,_,_ = read_video(opt.input_video, output_format="TCHW")
    frames = []
    for i in range(len(video)):
        image = T.ToPILImage()(video[i])
        image = image.resize((opt.h, opt.w),  resample=Image.Resampling.LANCZOS)
        frame = image.convert('RGB')
        frames = frames + [frame]
    frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(opt.device)
    latents = encode_imgs(vae, frames, deterministic=True).to(torch.float16).to(opt.device)
    
    ######### DDIM latent ################################
    # set timesteps for ddim latent save point
    ddim_scheduler.set_timesteps(opt.save_steps)
    save_steps = ddim_scheduler.timesteps
    # reset timesteps for ddim inference
    ddim_scheduler.set_timesteps(opt.steps)
    cond = get_text_embeds(tokenizer, text_encoder, opt.source_prompt, "", opt.device)[1].unsqueeze(0)
    ddim_inversion(cond, latents, ddim_scheduler, unet, save_path, save_steps)
    print("DDIM inversion finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config_path = 'configs/config_sample3.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # input: from configuration
    parser.add_argument('--device', type=str, default=config['device']) 
    parser.add_argument('--input_video', type=str, default=config['input_video']) 
    parser.add_argument('--h', type=int, default=config['h'])
    parser.add_argument('--w', type=int, default=config['w'])
    parser.add_argument('--save_steps', type=int, default=config['n_timesteps'])
    parser.add_argument('--source_prompt', type=str, default=config['source_prompt'])
    # output
    parser.add_argument('--out_dir', type=str, default='initial_latents')
    parser.add_argument('--steps', type=int, default=500)
    opt = parser.parse_args()
    print(f"source_prompt: {opt.source_prompt}")
    opt.video_name = Path(opt.input_video).stem
    run(opt)
