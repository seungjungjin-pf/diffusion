import copy
import math
import torch
from PIL import Image
import numpy as np


from matplotlib import pyplot as plt
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from text_embed import get_precomputed_tensors
from controlnext import ControlNeXtModel
from sd3 import SD3CNModel, get_sigmas
from diffusers.optimization import get_scheduler

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def generate_image(transformer, vae, scheduler, text_embeddings, pooled_text_embeddings, epoch, device, batch_size=1, num_inference_steps=1000):
    num_channels_latents = transformer.config.in_channels
    height, width = 64, 64  # Latent space dimensions (depends on your model)
    
    # Generate random latent noise
    latents = torch.randn((batch_size, num_channels_latents, height, width))
    latents = latents.to(transformer.device)
    
    # Set number of inference steps
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=num_inference_steps)
    timesteps = timesteps.to(device)
    
    print("Generating image..")
    
    # Iteratively denoise latents
    for index, t in enumerate(timesteps):
        
        if (index + 1) % (num_inference_steps / 10) == 0:
            print(t)
            # Decode latents to an image
            with torch.no_grad():
                # latents = 1 / 0.18215 * latents
                image = vae.decode(latents).sample
            
            # Post-process image
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            
            # Convert the numpy array to a PIL image
            pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))
            pil_image.save(f"gen_images/{epoch}_{index}.png")
        
        # Predict the noise residual
        with torch.no_grad():
            noise_pred = transformer(latents, timestep=t.view(-1), encoder_hidden_states=text_embeddings, pooled_projections=pooled_text_embeddings).sample
    
        # Compute the previous noisy sample
        latents = scheduler.step(noise_pred, t.view(-1), latents).prev_sample 
        

def train():
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
    
    gen_scheduler = FlowMatchEulerDiscreteScheduler()
    
    learning_rate = 1e-4
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-4
    adam_epsilon = 1e-8
    
    data_list = get_precomputed_tensors(device=device)
    data_list = [data_list[0]]
            
    gradient_accumulation_steps = 1
    num_train_epochs = 10
    num_update_steps_per_epoch = math.ceil(len(data_list) / gradient_accumulation_steps)
    
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
    
    
    # Disable gradient computation for VAE and text embedders
    for param in vae.parameters():
        param.requires_grad = False

    control_next = ControlNeXtModel(upscale_dim=1536).to(device)
    
    # Optimizer creation
    # params_to_optimize = control_next.parameters()
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

    first_epoch = 0
    num_train_epochs = 1000
    loss_list = []
    for epoch in range(first_epoch, num_train_epochs):
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

            res = control_next(hint_values, timesteps)
            control_out = res['output']
           
            # print(noisy_model_input.device) 
            # print(timesteps.device)
            # print(prompt_embeds.device)
            # print(pooled_prompt_embeds.device)
            # print(control_out.device)

            # import pdb; pdb.set_trace()

            # Predict the noise residual
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                controlnet_hidden_states=control_out,
                return_dict=False,
            )[0]

            # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
            # Preconditioning of the model outputs.
            # if args.precondition_outputs:
            #     model_pred = model_pred * (-sigmas) + noisy_model_input

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)

            # flow matching loss
            # if args.precondition_outputs:
            #     target = model_input
            # else:
            target = noise - model_input

            # Compute regular loss.
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            loss_list.append(loss.item())
            lr_scheduler.step()
            optimizer.zero_grad()
           
        if epoch % 10 == 0:
            print(np.mean(loss_list[10:]))
            generate_image(transformer, vae, gen_scheduler, prompt_embeds, pooled_prompt_embeds, epoch, device)

if __name__ == '__main__':
    train()