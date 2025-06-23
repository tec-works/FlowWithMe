# Align Your Flow: Text-to-Image Distillation

This repository contains the official PyTorch implementation for "Align Your Flow" (arXiv:2506.14603v1), focusing on the advanced text-to-image distillation task. The `train_ayf.py` script is designed to replicate the paper's methodology for distilling large-scale models like FLUX.1 using the Hugging Face ecosystem.

## Key Features

This implementation is aligned with the paper and includes:

-   **State-of-the-Art Teacher Model**: Utilizes the powerful **FLUX.1** text-to-image transformer from Hugging Face as the teacher model for distillation.
-   **Large-Scale Data Handling**: Employs the `webdataset` library to efficiently stream and process large, web-scale datasets like `text-to-image-2M`.
-   **Modern Training Framework**: Integrated with **Hugging Face `accelerate`** for seamless multi-GPU and mixed-precision training.
-   **Paper-Aligned Algorithms**:
    -   **AYF-EMD Loss**: The core distillation objective (Algorithm 1) to train the student flow map.
    -   **Autoguidance**: Distillation from an autoguided teacher to enhance sample quality.
    -   **Adversarial Finetuning (Algorithm 2)**: An optional second stage using a **StyleGAN2-inspired Discriminator** and the stable **RpGAN loss** with R1/R2 regularization.

## Requirements

To run the training script, you will need a multi-GPU environment with the following packages installed. You can install them using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

The key dependencies are:
- `torch` & `torchvision`
- `diffusers`
- `accelerate`
- `transformers`
- `webdataset`
- `tqdm`
- `matplotlib`
- `pyyaml`

## Setup & Training

The training process is controlled by a YAML configuration file.

### 1. Configure Your Training Run

Create a `config.yaml` file to specify all hyperparameters. An example configuration is provided in `configs/ayf_config.yaml`. You will need to adjust paths and parameters according to your setup.

**Example `config.yaml`:**
```yaml
# Model paths (from Hugging Face Hub)
model:
  teacher_model_id: "black-forest-labs/FLUX.1-dev"
  autoguide_model_id: "black-forest-labs/FLUX.1-schnell"

# Dataset configuration
data:
  name: "text-to-image-2M"
  urls: "pipe:curl -L -s [https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data](https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data)_{000000..000000}.tar"
  num_samples: 42000 
  resolution: 512
  num_workers: 4

# Training parameters
train:
  output_dir: "./ayf-training-output"
  num_epochs: 50
  batch_size_per_gpu: 8
  gradient_accumulation_steps: 4
  lr_student: 1.0e-4
  lr_discriminator: 2.0e-5
  mixed_precision: "bf16"
  adversarial_start_epoch: 40
  adv_loss_weight: 0.1
  r1_reg_weight: 0.1

# AYF-specific loss parameters
ayf_loss:
  p_mean: -0.8
  p_std: 1.0
  warmup_iters: 10000
  tangent_norm_c: 0.1
  autoguide_weight: 2.0
```

### 2. Launch Training

Use `accelerate` to launch the distributed training process.

```bash
accelerate launch train_ayf.py --config /path/to/your/config.yaml
```

The script will handle the two-stage training automatically:
1.  **Distillation Phase**: Trains the student using the AYF-EMD loss until the `adversarial_start_epoch` is reached.
2.  **Finetuning Phase**: Activates the discriminator and uses the combined EMD and adversarial losses to complete the training.

## Generating Images

Once a model is trained, use `generate.py` to create images from a checkpoint.

```bash
python generate.py --checkpoint /path/to/checkpoint/dir/pytorch_model.bin --outdir ./output --num-images 16
```

The generation script uses the stochastic `y-sampling` (gamma-sampling) method described in the paper.

