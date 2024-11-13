import copy
import math
import torch
from PIL import Image
import numpy as np
import os


from matplotlib import pyplot as plt
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from text_embed import get_precomputed_tensors
from controlnext import ControlNeXtModel
from sd3 import SD3CNModel, get_sigmas
from diffusers.optimization import get_scheduler

from torch.utils.tensorboard import SummaryWriter

from train_utils import generate_image
from sd3_pipeline import SD3CNPipeline

        
def train(base_log_dir="logs/sd3_training", run_type=None, index_block_location=0, gen_image_every=100, num_train_epochs=6000):
    # Increment run number until a new directory is found
    run_number = 0
    run_dir = f"{run_number}"
    if run_type is not None:
        run_dir = f"{run_type}_{run_dir}"
    while os.path.exists(os.path.join(base_log_dir, run_dir)):
        run_number += 1
        run_dir = f"{run_number}"
        if run_type is not None:
            run_dir = f"{run_type}_{run_dir}"

    log_dir = os.path.join(base_log_dir, run_dir)
    print("Logging to:", log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    weight_dtype = torch.float32
    logit_mean = 0.0
    logit_std = 1.0
    mode_scale = 1.29
    weighting_scheme = "logit_normal"
    device = 'cuda:0'
  
    scheduler_type = "constant"
    lr_warmup_steps = 500
    max_train_steps = 1000
    lr_num_cycles = 1
    lr_power = 1.0
    
    learning_rate = 1e-4
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-4
    adam_epsilon = 1e-8
    
    data_list = get_precomputed_tensors(device=device)
    data_list = [data_list[0]]
            
    transformer = SD3CNModel.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        subfolder="transformer", 
        # torch_dtype=torch.float16,
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            subfolder="vae",
            revision="refs/pr/26",
    ).to(device)
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    pipe = SD3CNPipeline(transformer, noise_scheduler, vae, device)
    
    # Disable gradient computation for VAE and text embedders
    for param in vae.parameters():
        param.requires_grad = False

    control_next = ControlNeXtModel(upscale_dim=1536).to(device)
    
    # Optimizer creation
    params_to_optimize = list(transformer.parameters()) + list(control_next.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    ) 

    for epoch in range(num_train_epochs):
        for step, data in enumerate(data_list):
            # Convert images to latent space
            pixel_values = data['img']
            hint_values = data['hint']
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
            model_input = model_input.to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=bsz,
                logit_mean=logit_mean,
                logit_std=logit_std,
                mode_scale=mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(timesteps, noise_scheduler_copy, n_dim=model_input.ndim, dtype=model_input.dtype, device=model_input.device)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            # Get the text embedding for conditioning
            prompt_embeds = data["prompt_embeds"].to(device)
            pooled_prompt_embeds = data["pooled_prompt_embeds"].to(device)

            # controlnet(s) inference
            # use_controlnext = np.random.rand() < 0.5
            control_hidden_states = control_next(hint_values, timesteps)['output']
           
            # Predict the noise residual
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                control_hidden_states=control_hidden_states,
                return_dict=False,
                index_block_location=index_block_location
            )[0]

            # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
            # Preconditioning of the model outputs.
            # if args.precondition_outputs:
            #     model_pred = model_pred * (-sigmas) + noisy_model_input

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)

            target = noise - model_input

            # Compute regular loss.
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            writer.add_scalar('Loss', loss.item(), epoch)
            lr_scheduler.step()
            optimizer.zero_grad()
           
    if epoch % gen_image_every == 0 and epoch != 0:
        img = generate_image(pipe, prompt_embeds, pooled_prompt_embeds, control_hidden_states, index_block_location)
        writer.add_image('Image', img, epoch)

if __name__ == '__main__':
    for i in range(24):
        print("Current block location:", i)
        train(base_log_dir='logs/sd3-img-gen-fixed', run_type=f"emb_{i}", index_block_location=i, gen_image_every=100, num_train_epochs=2000)