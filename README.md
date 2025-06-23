# Align Your Flow: A PyTorch Implementation

This repository provides a comprehensive and self-contained PyTorch implementation of the "Align Your Flow" paper (arXiv:2506.14603v1). The script is designed to be a faithful, conceptual reproduction of the paper's core algorithms for research and educational purposes.

It consolidates all major features into a single, runnable Python file, `main.py`, which uses mock U-Net models and `torchvision.datasets.FakeData` to focus purely on the algorithmic details without external dependencies.

## Key Features

This implementation includes all the critical components discussed in the paper:

-   **Flow Map Distillation**: The core concept of distilling a large teacher model into an efficient few-step student model that learns a "flow map" between any two noise levels.
-   **AYF-EMD Loss (Algorithm 1)**: A full implementation of the AYF-Eulerian Map Distillation loss, including the numerically stable tangent calculation using a Jacobian-vector product (`torch.func.jvp`).
-   **Training Stabilization (Section 3.4)**:
    -   **Tangent Normalization**: Regularizes the tangent vector to prevent training instability.
    -   **Regularized Tangent Warmup**: A curriculum strategy that gradually introduces the tangent-matching objective.
-   **Autoguidance (Section 3.3)**: Distillation from an "autoguided" teacher, where a high-quality model is guided by a weaker version of itself to improve sample quality while preserving diversity.
-   **Adversarial Finetuning (Algorithm 2)**: An optional second training stage to further enhance perceptual quality, featuring:
    -   A faithful **StyleGAN2-inspired Discriminator**.
    -   The stable **Relativistic Pairing GAN (RpGAN)** loss.
    -   **R1/R2 Gradient Penalty** on both real and fake images for stabilization.
    -   **Adaptive Weighting** to dynamically balance the EMD and adversarial losses.
-   **γ-Sampling (Section 5)**: A stochastic multi-step sampling algorithm (gamma-sampling) for inference, allowing for a flexible trade-off between speed and quality.

## How It Works

The script is structured around the two-stage training process described in the paper:

1.  **Stage 1: Distillation (Algorithm 1)**
    -   The `AlignYourFlow` model, wrapping a student U-Net, is trained to mimic the flow of a pre-trained (mock) teacher model.
    -   The training is driven by the `get_ayf_emd_loss` function, which calculates the core distillation loss.
    -   This stage focuses on teaching the student the fundamental structure of the data distribution.

2.  **Stage 2: Adversarial Finetuning (Algorithm 2)**
    -   The distilled student model from Stage 1 is further finetuned against a StyleGAN2-based discriminator.
    -   This stage uses a combined loss function: the EMD loss acts as a regularizer, while the adversarial (RpGAN) loss pushes the generator to produce more perceptually realistic images.

## Requirements

To run this script, you will need a Python environment with PyTorch and a few common libraries.

```bash
pip -r requirements.txt
```

## How to Run

The script is self-contained and can be run directly from the command line:

```bash
python main.py
```

The script will:

1.  Initialize the mock models and the `FakeData` loader.
2.  Run the Stage 1 distillation process, printing progress.
3.  Run the Stage 2 adversarial finetuning process, printing progress.
4.  The current implementation is set for demonstration and does not save image outputs, but the `sample` method and visualization logic can be easily adapted to do so.

## Code Structure

-   **`MockUnet`**: A placeholder for a U-Net architecture (like EDM2) used for the student, teacher, and weak teacher models.
-   **`StyleGAN2Discriminator`**: A faithful implementation of the StyleGAN2 discriminator architecture.
-   **`AlignYourFlow` Class**: The main class that orchestrates the training, sampling, and loss calculations, tying all the components together.

