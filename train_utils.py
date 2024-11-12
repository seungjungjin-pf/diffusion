import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union
import inspect


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
            epoch = str(epoch+1).zfill(4)
            index = str(index+1).zfill(4)
            pil_image.save(f"gen_images/{epoch}_{index}.png")
        
        # Predict the noise residual
        with torch.no_grad():
            noise_pred = transformer(latents, timestep=t.view(-1), encoder_hidden_states=text_embeddings, pooled_projections=pooled_text_embeddings).sample
    
        # Compute the previous noisy sample
        latents = scheduler.step(noise_pred, t.view(-1), latents).prev_sample 