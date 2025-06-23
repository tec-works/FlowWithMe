"""
sampler.py

This module provides the inference logic for a trained Align Your Flow model,
implementing the stochastic multi-step y-sampling (gamma-sampling) algorithm
from the paper.
"""
import torch
import math
from tqdm.auto import tqdm

@torch.no_grad()
def sample(ayf_model, scheduler, shape, text_embeds, num_steps, device, guidance_lambda, gamma):
    """
    Generates a batch of samples using y-sampling (gamma-sampling).

    Args:
        ayf_model (nn.Module): The trained AlignYourFlow model.
        scheduler (DDPMScheduler): The noise scheduler used during training.
        shape (tuple): The shape of the output tensor (B, C, H, W).
        text_embeds (torch.Tensor): Pre-computed text embeddings for conditioning.
        num_steps (int): The number of inference steps (NFE).
        device (torch.device): The device to run inference on.
        guidance_lambda (float): The autoguidance scale.
        gamma (float): The stochasticity parameter for sampling (in [0, 1]).

    Returns:
        torch.Tensor: A batch of generated images in the [0, 1] range.
    """
    ayf_model.eval()
    
    # Start with pure Gaussian noise
    latents = torch.randn(shape, device=device)
    
    # Set the timesteps for the scheduler
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    
    lambda_tensor = torch.full((shape[0],), guidance_lambda, device=device)

    for i in tqdm(range(num_steps), desc="y-Sampling"):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1] if i < num_steps - 1 else torch.tensor(0, device=device)
        
        # Reshape t and s for the model
        t_curr_tensor = t_curr.expand(shape[0])
        t_next_tensor = t_next.expand(shape[0])
        
        # 1. Denoising Jump: Use the flow map for a direct jump
        x_next_pred = ayf_model.flow_map_prediction(
            latents, t_curr_tensor, t_next_tensor, text_embeds, lambda_tensor
        )
        
        # 2. Stochastic Injection (if not the last step)
        if gamma > 0 and t_next > 0:
            # Noise variance is scaled by the change in sigma^2
            # For a scheduler, this variance can be retrieved directly
            variance = scheduler.get_variance(t_curr, t_next)
            noise_std = gamma * torch.sqrt(variance)
            latents = x_next_pred + torch.randn_like(x_next_pred) * noise_std
        else:
            latents = x_next_pred
            
    # Denormalize from [-1, 1] to [0, 1] for saving/visualization
    images = (latents.clamp(-1, 1) + 1) / 2
    return images
