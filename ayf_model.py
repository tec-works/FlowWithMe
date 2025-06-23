"""
ayf_model.py

This module contains the PyTorch nn.Module definitions for the key
components of the Align Your Flow model, including:
- The main AlignYourFlow student model wrapper.
- The StyleGAN2-inspired discriminator for adversarial finetuning.
"""
import torch
import torch.nn as nn
import math
from diffusers.models.transformers.flux import FLUXTransformer2DModel

# --- Discriminator Components ---

class Downsample(nn.Module):
    """ Simple average pooling downsampler. """
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        return self.pool(x)

class ResBlock(nn.Module):
    """ Residual block for the StyleGAN2 discriminator. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.downsample = Downsample()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        skip_x = self.downsample(self.skip(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.downsample(x)
        # The scaling factor sqrt(2) is used to preserve variance
        return (skip_x + x) * (1 / math.sqrt(2))

class MinibatchStdDevLayer(nn.Module):
    """ Minibatch Standard Deviation layer from StyleGAN2. """
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        N, C, H, W = x.shape
        # Ensure group_size is not larger than the batch size
        group_size = min(N, self.group_size)
        y = x.view(group_size, -1, self.num_new_features, C // self.num_new_features, H, W)
        y = y - y.mean(0, keepdim=True)
        y = y.square().mean(0)
        y = (y + 1e-8).sqrt()
        y = y.mean([2, 3, 4], keepdim=True).squeeze(0)
        y = y.repeat(group_size, 1, H, W)
        return torch.cat([x, y], dim=1)

class StyleGAN2Discriminator(nn.Module):
    """ A faithful PyTorch implementation of the StyleGAN2 discriminator architecture. """
    def __init__(self, resolution, in_channels=3):
        super().__init__()
        channels = {
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256,
            128: 128, 256: 64, 512: 32, 1024: 16
        }
        log2_res = int(math.log2(resolution))
        
        self.from_rgb = nn.Conv2d(in_channels, channels[resolution], 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        blocks = []
        for i in range(log2_res, 2, -1):
            res = 2**i
            in_ch, out_ch = channels[res], channels[res // 2]
            blocks.append(ResBlock(in_ch, out_ch))
        self.blocks = nn.ModuleList(blocks)
        
        self.mbstd = MinibatchStdDevLayer()
        final_ch = channels[4]
        self.final_conv = nn.Conv2d(final_ch + 1, final_ch, 3, padding=1)
        self.final_dense = nn.Linear(final_ch * 4 * 4, final_ch)
        self.output_layer = nn.Linear(final_ch, 1)

    def forward(self, x):
        x = self.activation(self.from_rgb(x))
        for block in self.blocks:
            x = block(x)
        x = self.mbstd(x)
        x = self.activation(self.final_conv(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.final_dense(x))
        return self.output_layer(x)

# --- AYF Model Wrapper ---

class AlignYourFlow(nn.Module):
    """ 
    Orchestrates the AYF student model, wrapping the core transformer and
    providing the methods for loss calculation and inference.
    """
    def __init__(self, student_transformer, teacher_transformer, weak_teacher_transformer):
        super().__init__()
        self.student_net = student_transformer
        self.teacher_net = teacher_transformer
        self.weak_teacher_net = weak_teacher_transformer
        # Teachers do not require gradients
        for p in self.teacher_net.parameters(): p.requires_grad = False
        for p in self.weak_teacher_net.parameters(): p.requires_grad = False

    def get_teacher_velocity(self, x_t, t, text_embeds, guidance_scale):
        """ Calculates the autoguided teacher velocity (Eq. 3). """
        strong_v = self.teacher_net(hidden_states=x_t, timestep=t, text_embeds=text_embeds).sample
        weak_v = self.weak_teacher_net(hidden_states=x_t, timestep=t, text_embeds=text_embeds).sample
        return (1 + guidance_scale) * strong_v - guidance_scale * weak_v

    def F_theta(self, x_t, t, s, text_embeds, lambda_val):
        """ The core student network call. """
        # NOTE: A true AYF model might condition F_theta on s and lambda_val.
        # Here we follow the practical approach of using a standard transformer.
        return self.student_net(hidden_states=x_t, timestep=t, text_embeds=text_embeds).sample

    def flow_map_prediction(self, x_t, t, s, text_embeds, lambda_val):
        """ Computes f_theta(x_t, t, s) via Euler parameterization (Sec 3.2). """
        F_theta_output = self.F_theta(x_t, t, s, text_embeds, lambda_val)
        time_diff = (s - t).view(-1, 1, 1, 1)
        return x_t + time_diff * F_theta_output
