# Denoising Diffusion Probabilistic Models (DDPM)

This directory contains a complete implementation of Denoising Diffusion Probabilistic Models, demonstrating how to generate high-quality images through iterative denoising.

> This section was created by [Angel Aaron Flores Alberca](https://github.com/bxcowo) active member of the [NONHUMAN](https://nonhuman.site) organization.

## Contents

### 📓 Notebook

#### `DDPM.ipynb`
**Complete DDPM Implementation on MNIST**

A from-scratch implementation of the DDPM paper (Ho et al., 2020) trained on MNIST digits. This notebook demonstrates how diffusion models progressively add and remove noise to generate images.

### 📄 Theory Document

#### `DDPM.pdf`
Theoretical foundations of Denoising Diffusion Probabilistic Models, including mathematical derivations and a few of the algorithm details.

## What You'll Learn

The notebook covers the complete DDPM pipeline:

1. **Forward Diffusion Process**
   - Linear noise schedule (β₁ to βₜ)
   - Direct sampling at any timestep using reparameterization
   - Visual demonstration of progressive noise corruption (t=0 to t=T)

2. **U-Net Architecture for Noise Prediction**
   - Encoder-decoder structure with skip connections
   - Sinusoidal time embeddings (similar to Transformers)
   - Adaptive Group Normalization for time conditioning
   - ResNet blocks with time injection

3. **Training Process**
   - Simplified DDPM objective: predict noise ε
   - Random timestep sampling during training
   - Loss function implementation (MSE between predicted and actual noise)

4. **Reverse Diffusion Process (Sampling)**
   - Iterative denoising from pure noise to image
   - Posterior variance calculation
   - Step-by-step image generation over T timesteps

5. **Visualization**
   - Forward process visualization (clean → noisy)
   - Training progress monitoring
   - Sample generation from random noise

## Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib tqdm
```

### Hardware Requirements

**Minimum:**
- GPU with 6GB+ VRAM (CUDA-capable)
- 8GB RAM

**Recommended:**
- GPU with 8GB+ VRAM
- 16GB RAM

**Training time:** ~30-60 minutes for reasonable quality on MNIST (15 epochs)

### Configuration

Key hyperparameters in the notebook:

```python
T = 300              # Number of diffusion timesteps
BETA_START = 1e-4    # β₁ (minimum noise)
BETA_END = 0.02      # βₜ (maximum noise)
batch_size = 128
device = "cuda:0"    # Change to "cpu" if no GPU
```

### Running the Notebook

1. **Noise Schedule Setup**
   - Defines linear schedule of βₜ values
   - Precomputes αₜ and ᾱₜ for efficiency
   - Visualizes noise corruption at different timesteps

2. **Model Architecture**
   - U-Net with time conditioning
   - 4.9M parameters
   - Encoder: 1×28×28 → 64×28×28 → 128×14×14 → 256×7×7
   - Decoder: 256×7×7 → 128×14×14 → 64×28×28 → 1×28×28

3. **Training**
   - Runs for 15 epochs on MNIST
   - Training objective: predict noise ε from noisy image xₜ
   - Progress tracked via loss curves

4. **Sampling**
   - Generate new digits from pure Gaussian noise
   - 300-step iterative denoising process
   - Visualize intermediate denoising steps

## Model Architecture Details

### Forward Process (Diffusion)

The forward process gradually adds Gaussian noise according to:

```
q(xₜ | x₀) = N(xₜ; √ᾱₜ · x₀, (1 - ᾱₜ) · I)
```

**Direct sampling** at any timestep t:
```
xₜ = √ᾱₜ · x₀ + √(1 - ᾱₜ) · ε,  where ε ~ N(0, I)
```

This allows jumping directly to any timestep without simulating intermediate steps.

### U-Net with Time Conditioning

**Time Embedding:**
- Sinusoidal positional encoding (same as Transformers)
- MLP projection to time_dim (256)
- Injected into each ResBlock via Adaptive Group Normalization

**Architecture:**
```
Input: xₜ (noisy image) + t (timestep)

Encoder:
  ResBlock(1→64) → Downsample → 64×28×28
  ResBlock(64→128) → Downsample → 128×14×14
  ResBlock(128→256) → 256×7×7

Bottleneck:
  ResBlock(256→256)

Decoder (with skip connections):
  Upsample + Concat + ResBlock(256→128) → 128×14×14
  Upsample + Concat + ResBlock(128→64) → 64×28×28
  Conv1x1 → 1×28×28

Output: Predicted noise ε_θ(xₜ, t)
```

### Training Objective

**Simplified DDPM loss** (Ho et al., 2020):

```
L = E[||ε - ε_θ(xₜ, t)||²]
```

Where:
- ε is the actual noise added to x₀
- ε_θ(xₜ, t) is the U-Net's prediction
- xₜ = √ᾱₜ · x₀ + √(1 - ᾱₜ) · ε

Algorithm:
1. Sample x₀ from training data
2. Sample t uniformly from [1, T]
3. Sample noise ε ~ N(0, I)
4. Compute xₜ using direct sampling
5. Predict ε̂ = ε_θ(xₜ, t)
6. Minimize ||ε - ε̂||²

### Reverse Process (Sampling)

Starting from xₜ ~ N(0, I), iteratively denoise:

```
xₜ₋₁ = (1/√αₜ) · (xₜ - (βₜ/√(1-ᾱₜ)) · ε_θ(xₜ, t)) + σₜ · z
```

Where:
- z ~ N(0, I) for t > 1, z = 0 for t = 1
- σₜ² = βₜ · (1 - ᾱₜ₋₁) / (1 - ᾱₜ) (posterior variance)

This process runs for all T timesteps to generate a clean sample.

## Expected Results

After training (15 epochs):

- **Training loss:** Converges to ~0.02-0.03
- **Generated samples:** Recognizable MNIST digits
- **Sampling time:** ~30 seconds for 300 steps on GPU
- **Model size:** ~20MB checkpoint

## Key Differences from VAEs

| Aspect | VAE | DDPM |
|--------|-----|------|
| **Generation** | 1-step (decoder) | 300+ steps (iterative) |
| **Latent space** | Learned encoding | Fixed noise schedule |
| **Quality** | Often blurry | Sharp, high-quality |
| **Speed** | Fast (~1ms) | Slow (~30s for T=300) |
| **Training** | Encoder + Decoder | Noise predictor only |

**Why DDPM?**
- No posterior collapse
- Better mode coverage
- State-of-the-art sample quality
- Flexible conditioning (class, text, etc.)

## Experiments to Try!

1. **Noise Schedule Variations**
   - Try cosine schedule instead of linear
   - Adjust BETA_START and BETA_END
   - Experiment with different T values (100, 500, 1000)

2. **Faster Sampling**
   - DDIM sampling (skip timesteps, ~10x speedup)
   - Implement stride sampling (every 5th step)
   - Compare quality vs speed trade-offs

3. **Conditional Generation**
   - Add class labels to U-Net
   - Generate specific digits on demand
   - Classifier-free guidance

4. **Architecture Modifications**
   - Add attention layers in bottleneck
   - Increase model capacity (more channels)
   - Self-attention at multiple resolutions

5. **Other Datasets**
   - CIFAR-10 (32×32 RGB)
   - Fashion-MNIST
   - CelebA faces (requires larger model)


## Further Reading

- **Original Paper:** [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- **DDIM:** [Denoising Diffusion Implicit Models (Song et al., 2020)](https://arxiv.org/abs/2010.02502)
- **Score-based:** [Score-Based Generative Modeling (Song & Ermon, 2019)](https://arxiv.org/abs/1907.05600)
- **Stable Diffusion:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## Latent Diffusion (Stable Diffusion)

Modern diffusion models like Stable Diffusion combine:
- **VAE:** Compress images to latent space (4-8× smaller)
- **DDPM:** Run diffusion in compressed latent space
- **Transformer:** Text conditioning via CLIP embeddings

This reduces computational cost while maintaining quality.

---

**Note:** This implementation prioritizes clarity and education. Production models (Stable Diffusion, DALL-E) use more sophisticated architectures, larger datasets, and extensive hyperparameter tuning.
