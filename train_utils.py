import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union
import inspect
from diffusers import FlowMatchEulerDiscreteScheduler
import torch.nn.functional as F


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


def generate_image(pipe, prompt_embeds, pooled_prompt_embeds, control_next_model, control_input, index_block_location, height=512, width=512, use_controlnext=True):
    print("####Generating image")
    
    if use_controlnext: 
        res = pipe(prompt_embeds=prompt_embeds, 
                pooled_prompt_embeds=pooled_prompt_embeds,
                control_next_model=control_next_model,
                control_input=control_input,
                index_block_location=index_block_location,
                height=height,
                width=width,
                guidance_scale=0.5)
    else:
        res = pipe(prompt_embeds=prompt_embeds, 
                pooled_prompt_embeds=pooled_prompt_embeds,
                control_next_model=None,
                control_input=None,
                index_block_location=index_block_location,
                height=height,
                width=width,
                guidance_scale=0.5)
    image_list = res.images
    image_list = [torch.tensor(np.asarray(image)).permute(2,0,1) for image in image_list]
    print("####Done generating image")
    
    # Display the final grid image
    # return torch.tensor(np.asarray(image)).permute(2,0,1)
    return image_list


# def cross_norm1(x_m, x_c, scale=0.1, control_scale=0.2):
def cross_norm1(x_m, x_c, scale=0.8, control_scale=0.8):
    """Implementation from ControlNeXt github"""
    mean_latents, std_latents = torch.mean(x_m, dim=(1, 2), keepdim=True), torch.std(x_m, dim=(1, 2), keepdim=True)
    mean_control, std_control = torch.mean(x_c, dim=(1, 2), keepdim=True), torch.std(x_c, dim=(1, 2), keepdim=True)
    
    conditional_controls = (x_c - mean_control) * (std_latents / (std_control + 1e-5)) + mean_latents
    conditional_controls = F.adaptive_avg_pool2d(conditional_controls, x_m.shape[-2:])

    return x_m + conditional_controls * scale * control_scale

def cross_norm2(x_m, x_c, scale=0.1, control_scale=0.2):
    """Implementation from paper"""
    mean_latents, std_latents = torch.mean(x_m, dim=(1, 2, 3), keepdim=True), torch.std(x_m, dim=(1, 2, 3), keepdim=True)
    mean_control, std_control = torch.mean(x_c, dim=(1, 2, 3), keepdim=True), torch.std(x_c, dim=(1, 2, 3), keepdim=True)
    
    conditional_controls = (x_c - mean_control) / torch.sqrt(torch.pow(std_control, 2) + 1e-5)
    conditional_controls = F.adaptive_avg_pool2d(conditional_controls, x_m.shape[-2:])

    return x_m + conditional_controls * scale


# def cross_norm1(x_m, x_c, scale=0.1, control_scale=0.2):
#     """Implementation from ControlNeXt github"""
#     mean_latents, std_latents = torch.mean(x_m, dim=(1, 2, 3), keepdim=True), torch.std(x_m, dim=(1, 2, 3), keepdim=True)
#     mean_control, std_control = torch.mean(x_c, dim=(1, 2, 3), keepdim=True), torch.std(x_c, dim=(1, 2, 3), keepdim=True)
    
#     conditional_controls = (x_c - mean_control) * (std_latents / (std_control + 1e-5)) + mean_latents
#     conditional_controls = F.adaptive_avg_pool2d(conditional_controls, x_m.shape[-2:])

#     return x_m + conditional_controls * scale * control_scale

# def cross_norm2(x_m, x_c, scale=0.1, control_scale=0.2):
#     """Implementation from paper"""
#     mean_latents, std_latents = torch.mean(x_m, dim=(1, 2, 3), keepdim=True), torch.std(x_m, dim=(1, 2, 3), keepdim=True)
#     mean_control, std_control = torch.mean(x_c, dim=(1, 2, 3), keepdim=True), torch.std(x_c, dim=(1, 2, 3), keepdim=True)
    
#     conditional_controls = (x_c - mean_control) / torch.sqrt(torch.pow(std_control, 2) + 1e-5)
#     conditional_controls = F.adaptive_avg_pool2d(conditional_controls, x_m.shape[-2:])

#     return x_m + conditional_controls * scale