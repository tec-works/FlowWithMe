"""
generate.py

This script loads a trained Align Your Flow (AYF) student model and uses it
to generate new images. It is designed to work with model checkpoints saved
from the `main.py` training script.

The script implements the stochastic `y-sampling` (gamma-sampling) algorithm
described in the paper for high-quality, multi-step inference.

Usage:
    python generate.py --checkpoint /path/to/student_net.pt --outdir ./output --num-images 16

"""
import torch
import torch.nn as nn
import math
import os
import argparse
import torchvision

# --- 1. Model Definitions ---
# These classes must match the definitions in `ayf_implementation.py` to ensure
# the model architecture can be correctly reconstructed from a checkpoint.

class FourierTimeEmbedding(nn.Module):
    """ Fourier feature time embedding to handle t, s, and lambda. """
    def __init__(self, embedding_dim: int):
        super().__init__()
        # Using register_buffer for non-trainable parameters
        self.register_buffer('freqs', torch.randn(embedding_dim // 2) * 2 * math.pi)

    def forward(self, t: torch.Tensor, s: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
        t_emb = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        s_emb = s.unsqueeze(-1) * self.freqs.unsqueeze(0)
        lambda_emb = lambda_val.unsqueeze(-1) * self.freqs.unsqueeze(0)
        t_emb_full = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        s_emb_full = torch.cat([torch.sin(s_emb), torch.cos(s_emb)], dim=-1)
        lambda_emb_full = torch.cat([torch.sin(lambda_emb), torch.cos(lambda_emb)], dim=-1)
        return t_emb_full + s_emb_full + lambda_emb_full

class ConvBlock(nn.Module):
    """ A basic convolutional block for the U-Net. """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.SiLU()
        )
    def forward(self, x): return self.net(x)

class MockUnet(nn.Module):
    """ A placeholder for the U-Net architecture (F_theta). """
    def __init__(self, in_channels=3, time_emb_dim=256, cond_emb_dim=512):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, cond_emb_dim)
        
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc1 = ConvBlock(64, 128)
        self.enc2 = ConvBlock(128, 256)
        self.pool = nn.AvgPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = ConvBlock(512, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.conv_out = nn.Conv2d(64, in_channels, 1)

    def forward(self, x, t, s, lambda_val):
        cond_emb = self.time_proj(self.time_embed(t, s, lambda_val)).unsqueeze(-1).unsqueeze(-1)
        x1 = self.conv_in(x)
        x2 = self.enc1(x1)
        x3 = self.pool(x2)
        x4 = self.enc2(x3)
        x5 = self.pool(x4)
        b = self.bottleneck(x5) + cond_emb
        d1 = self.up1(b); d1 = torch.cat([d1, x4], dim=1)
        d2 = self.dec1(d1)
        d3 = self.up2(d2); d3 = torch.cat([d3, x2], dim=1)
        d4 = self.dec2(d3)
        return self.conv_out(d4)

# --- 2. Inference Orchestrator ---

class AYFGenerator:
    """
    A simplified orchestrator for inference. It only needs the student network
    and encapsulates the sampling logic.
    """
    def __init__(self, student_net):
        self.student_net = student_net
        # Teacher models are not needed for inference
        self.teacher_net = None
        self.weak_teacher_net = None

    def flow_map_prediction(self, x_t, t, s, lambda_val):
        """ Computes f_theta(x_t, t, s) via Euler parameterization. """
        F_theta = self.student_net(x_t, t, s, lambda_val)
        time_diff = (s - t).view(-1, 1, 1, 1)
        return x_t + time_diff * F_theta

    @torch.no_grad()
    def sample(self, shape, num_steps, device, lambda_val=2.0, gamma=1.0):
        """ Performs multi-step y-sampling (gamma-sampling). """
        self.student_net.eval()
        x_t = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        lambda_tensor = torch.full((shape[0],), lambda_val, device=device)

        print(f"Generating with {num_steps} steps (gamma={gamma}, guidance={lambda_val})...")
        for i in range(num_steps):
            t_curr, t_next = ts[i], ts[i+1]
            t_curr_tensor = t_curr.expand(shape[0])
            t_next_tensor = t_next.expand(shape[0])
            
            x_next_pred = self.flow_map_prediction(x_t, t_curr_tensor, t_next_tensor, lambda_tensor)
            
            if gamma > 0 and t_next > 1e-6:
                noise_std = gamma * torch.sqrt(t_curr**2 - t_next**2)
                x_t = x_next_pred + torch.randn_like(x_next_pred) * noise_std.view(-1, 1, 1, 1)
            else:
                x_t = x_next_pred
        # Denormalize from [-1, 1] to [0, 1] for saving
        return (x_t + 1) / 2

# --- 3. Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description="Generate images with a trained Align Your Flow model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained student model checkpoint (.pt file).')
    parser.add_argument('--outdir', type=str, default='./output', help='Directory to save the generated images.')
    parser.add_argument('--num-images', type=int, default=16, help='Number of images to generate.')
    parser.add_argument('--steps', type=int, default=4, help='Number of sampling steps (NFE).')
    parser.add_argument('--gamma', type=float, default=0.9, help='Stochasticity parameter for y-sampling.')
    parser.add_argument('--guidance', type=float, default=2.0, help='Autoguidance scale lambda.')
    parser.add_argument('--resolution', type=int, default=64, help='Image resolution.')
    args = parser.parse_args()

    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    # --- Load Model ---
    print(f"Loading student model from {args.checkpoint}...")
    student_net = MockUnet().to(device)
    
    try:
        # Assumes the checkpoint contains the state_dict of the student model
        student_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}.")
        print("Please train a model first using `ayf_implementation.py` and save the student network's state_dict.")
        return
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        print("The checkpoint file might be corrupt or incompatible with the model architecture.")
        return
        
    ayf_generator = AYFGenerator(student_net)

    # --- Generate Images ---
    shape = (args.num_images, 3, args.resolution, args.resolution)
    samples = ayf_generator.sample(
        shape=shape,
        num_steps=args.steps,
        device=device,
        lambda_val=args.guidance,
        gamma=args.gamma
    )

    # --- Save Images ---
    output_path = os.path.join(args.outdir, f"ayf_s{args.steps}_g{args.gamma}_l{args.guidance}.png")
    torchvision.utils.save_image(samples, output_path, nrow=int(math.sqrt(args.num_images)))
    print(f"Saved {args.num_images} images to {output_path}")

if __name__ == '__main__':
    main()
