import copy
import numpy as np
import os
import torch
import torchvision

from argparse import ArgumentParser


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
from tqdm import tqdm

        
def train(base_log_dir="logs/sd3_training", run_type=None, use_controlnext=False, resize=False,
          index_block_location=0, gen_image_every=10, num_train_epochs=10, device='cuda:1', height=1024, width=1024,
          lr_scheduler_type='constant', dataset_name='prompt_tensors_list.pt', print_shapes=False):
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
  
    lr_warmup_steps = 500
    lr_num_cycles = 2
    lr_power = 1.0
    
    learning_rate = 1e-4
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-4
    adam_epsilon = 1e-8
    
    data_list = get_precomputed_tensors(device='cpu', filename=dataset_name)
    # data_list = [data_list[0]]
            
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

    if resize: 
        resize = torchvision.transforms.Resize((1024,1024))
    
    control_next = None
    
    # Optimizer creation
    params_to_optimize = list(transformer.parameters())
    if use_controlnext:
        control_next = ControlNeXtModel(upscale_dim=1536).to(device)
        params_to_optimize = list(transformer.parameters()) + list(control_next.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_data_list = data_list[:-2]
    test_data_list = data_list[-2:]
    for data in test_data_list:
        print(data['prompt'])
        
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=len(train_data_list)*num_train_epochs,
        num_cycles=num_train_epochs,
        power=lr_power,
    ) 
   
    global_step = 0 
    for epoch in range(num_train_epochs):
        with tqdm(train_data_list, total=len(train_data_list), desc=f"Epoch {epoch}") as pbar:
            for step, data in enumerate(pbar):
                if global_step % gen_image_every == 0:
                    transformer.eval()
                    for _i, data in enumerate(test_data_list):
                        for key, value in data.items():
                            if isinstance(value, torch.Tensor):
                                data[key] = data[key].to(device)
                        prompt_embeds = data["prompt_embeds"]
                        pooled_prompt_embeds = data["pooled_prompt_embeds"]
                        prompt = data["prompt"]
                        if control_next is not None and global_step != 0:
                            control_next.eval()
                            hint_img = data["hint"]
                            if resize:
                                hint_img = resize(hint_img).to(device)
                            control_input = vae.encode(hint_img).latent_dist.sample()
                            control_input = (control_input - vae.config.shift_factor) * vae.config.scaling_factor
                            control_input = control_input.to(dtype=weight_dtype)
                            img = generate_image(pipe, prompt_embeds, pooled_prompt_embeds, control_next, control_input, index_block_location, height=height, width=width)
                        else:
                            img = generate_image(pipe, prompt_embeds, pooled_prompt_embeds, None, None, index_block_location, height=height, width=width)
                        
                        writer.add_image(f'Image_{_i}_{prompt}', img, global_step)
                        
                        
                transformer.train()
                if control_next is not None:
                    control_next.train()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = data[key].to(device)
                # Convert images to latent space
                pixel_values = data['img']
                hint_values = data['hint']
                if resize:
                    pixel_values = resize(pixel_values).to(device)
                    hint_values = resize(hint_values).to(device)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
                
                control_input = vae.encode(hint_values).latent_dist.sample()
                control_input = (control_input - vae.config.shift_factor) * vae.config.scaling_factor
                control_input = control_input.to(dtype=weight_dtype)

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
                use_controlnext = np.random.rand() < 0.5
                control_hidden_states = None
                if control_next is not None and use_controlnext:
                    control_hidden_states = control_next(control_input, timesteps)['output']
            
                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    control_hidden_states=control_hidden_states,
                    return_dict=False,
                    index_block_location=index_block_location,
                    print_shapes=print_shapes,
                )[0]
                
                if global_step == 0 and print_shapes:
                    print("******************Training shapes******************")
                    print(f"Model input shape: {model_input.shape}")
                    print(f"Noisy model input shape: {noisy_model_input.shape}")
                    print(f"Model pred shape: {model_pred.shape}")
                    print(f"Prompt embeds shape: {prompt_embeds.shape}")
                    print(f"Pooled prompt embeds shape: {pooled_prompt_embeds.shape}")
                    if control_hidden_states is not None:
                        print(f"Control hidden states shape: {control_hidden_states.shape}")
                    

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
                optimizer.zero_grad()
                
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()
                
                writer.add_scalar('Loss', loss.item(), global_step)
                writer.add_scalar('lr', lr_scheduler.get_lr()[0], global_step)
                
                global_step += 1
                pbar.set_postfix(step=global_step, loss=loss.item())
        


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("--use-controlnext", action="store_true")
    args.add_argument("--print-shapes", action="store_true")
    args.add_argument("--gen-image-every", type=int, default=100)
    args.add_argument("--num-train-epochs", type=int, default=7000)
    args.add_argument("--lr-scheduler-type", type=str, default='constant')
    args.add_argument("--dataset-name", type=str, default='prompt_tensors_list.pt')
    args = args.parse_args()
    
    use_controlnext = args.use_controlnext
    index = 0
    train(base_log_dir='logs/sd3-large', run_type=f"use-control", index_block_location=index, 
          resize=True, lr_scheduler_type=args.lr_scheduler_type, dataset_name=args.dataset_name,
          use_controlnext=use_controlnext, gen_image_every=args.gen_image_every, num_train_epochs=args.num_train_epochs, 
          print_shapes=args.print_shapes,
          height=1024, width=1024)