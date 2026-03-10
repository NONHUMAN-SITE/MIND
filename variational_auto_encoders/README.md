# Variational Autoencoders (VAEs)

This directory contains educational materials on Variational Autoencoders (VAEs), including their implementation and limitations that motivate diffusion models.

> This section was created by [Angel Aaron Flores Alberca](https://github.com/bxcowo) active member of the [NONHUMAN](https://nonhuman.site) organization.

## Contents

### 📓 Notebook

#### `VAE.ipynb`
**Complete VAE Implementation on MNIST**

A practical, end-to-end implementation of a Variational Autoencoder trained on the MNIST dataset. This notebook demonstrates the core concepts of VAEs through a working example, and concludes with an explanation of why Denoising Diffusion Probabilistic Models (DDPM) became necessary.

## What You'll Learn

The notebook covers the complete VAE pipeline:

1. **VAE Architecture**
   - Convolutional encoder with batch normalization
   - Latent space parameterization (μ and log σ²)
   - Reparameterization trick for backpropagation
   - Convolutional decoder with transposed convolutions

2. **Loss Function**
   - Reconstruction loss (MSE)
   - KL divergence regularization
   - Combined ELBO (Evidence Lower Bound) objective

3. **Training Process**
   - MNIST dataset preparation and normalization
   - Training loop implementation
   - Model checkpointing

4. **Inference and Visualization**
   - Image reconstruction from test data
   - Latent space sampling and generation
   - Visual comparison of original vs reconstructed images

5. **Limitations and Next Steps**
   - Understanding VAE limitations (blurry outputs, posterior constraints)
   - Why Denoising Diffusion Probabilistic Models (DDPM) were developed
   - Trade-offs between VAEs and diffusion models

## Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib
```

### Hardware Requirements

**Minimum:**
- GPU with 4GB+ VRAM (CUDA-capable)
- 8GB RAM

**CPU-only:** Possible but significantly slower

### Configuration

Key hyperparameters defined in the notebook:

```python
batch_size = 32
latent_dim = 256      # Dimensionality of latent space
device = "cuda:0"     # Change to "cpu" if no GPU available
num_epochs = 15
```

### Running the Notebook

1. **Setup and Data Loading**
   - Installs dependencies
   - Downloads MNIST dataset automatically
   - Configures normalization (mean=0.1307, std=0.3081)

2. **Model Architecture**
   - **Encoder**: 3 convolutional layers (1→32→64→128 channels) with stride=2 downsampling
   - **Latent Layer**: Splits into μ and log σ² branches
   - **Decoder**: 3 transposed convolutional layers (128→64→32→1) for reconstruction

3. **Training**
   - Run the training cell to train for 15 epochs
   - Expected final loss: ~315 (reconstruction + KL)
   - Training time: ~5-10 minutes on modern GPU

4. **Model Saving**
   - Saves trained model as `vae_mnist.pth`

5. **Visualization**
   - Loads saved model
   - Shows 8 original images vs reconstructions
   - Generates 16 new samples from random latent vectors

## Model Architecture Details

### Encoder Path
```
Input: 28×28 grayscale image
Conv2d(1→32, k=3, s=2) → 14×14×32
Conv2d(32→64, k=3, s=2) → 7×7×64
Conv2d(64→128, k=3, s=2) → 4×4×128
Flatten → 2048-dim vector

Split into two branches:
- μ branch: Linear(2048→512→256)
- log σ² branch: Linear(2048→512→256)
```

### Reparameterization Trick
```
z = μ + ε * σ, where ε ~ N(0, I)
```
This allows backpropagation through the stochastic sampling operation.

### Decoder Path
```
Latent vector (256-dim)
Linear(256→2048) → Reshape to 4×4×128
ConvTranspose2d(128→64, k=3, s=2) → 7×7×64
ConvTranspose2d(64→32, k=3, s=2) → 14×14×32
ConvTranspose2d(32→1, k=3, s=2) → 28×28×1
Tanh activation
```

## Loss Function Breakdown

The VAE loss combines two terms:

**Reconstruction Loss:**
```
L_recon = MSE(x_reconstructed, x_original)
```
Encourages faithful reconstruction of inputs.

**KL Divergence:**
```
L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```
Regularizes latent space to be close to N(0, I), enabling smooth interpolation and sampling.

**Total Loss:**
```
L_total = L_recon + L_KL
```

## Expected Results

After training (15 epochs):

- **Reconstruction quality**: Sharp, recognizable digits
- **Generated samples**: Realistic-looking handwritten digits
- **Loss convergence**: From ~378 to ~315
- **Model size**: ~10MB (`vae_mnist.pth`)

## Experiments to Try

1. **Latent Dimension Exploration**
   - Try `latent_dim = 2` for 2D visualization
   - Try `latent_dim = 512` for higher capacity
   - Observe quality vs compression trade-off

2. **Loss Weighting**
   - Add β parameter: `L_total = L_recon + β * L_KL`
   - β < 1: Better reconstruction, less regularization
   - β > 1: More regularized latent space (β-VAE)

3. **Latent Space Interpolation**
   ```python
   z1 = torch.randn(1, latent_dim)
   z2 = torch.randn(1, latent_dim)
   # Interpolate between z1 and z2
   alphas = torch.linspace(0, 1, 10)
   for alpha in alphas:
       z_interp = (1-alpha)*z1 + alpha*z2
       # Decode and visualize
   ```

4. **Longer Training**
   - Increase `num_epochs` to 30-50
   - Monitor when loss plateaus
   - Compare reconstruction quality

## Common Issues

**CUDA Out of Memory:**
- Reduce `batch_size` to 16 or 8
- Decrease `latent_dim` to 128
- Use a smaller model (fewer channels)

**Poor Reconstruction:**
- Train for more epochs
- Increase model capacity (more channels/layers)
- Adjust β weighting in loss function

**Blurry Generated Images:**
- Normal for VAEs due to MSE loss
- Try different loss functions (perceptual loss, adversarial loss)
- Increase latent dimensionality

**Model not saved:**
- Ensure write permissions in directory
- Check disk space
- Run the save cell explicitly

## Key Concepts

**Variational Inference:**
VAEs learn an approximate posterior distribution q(z|x) that is close to the true posterior p(z|x).

**Reparameterization Trick:**
Makes the sampling operation differentiable by expressing it as deterministic transformation of input and noise.

**Latent Space:**
Low-dimensional representation where similar inputs are close together, enabling interpolation and controlled generation.

**ELBO (Evidence Lower Bound):**
The VAE loss is actually maximizing a lower bound on the log-likelihood of the data.

## Connection to Diffusion Models

The notebook concludes with a detailed explanation of VAE limitations and why DDPMs were developed:

**VAE Limitations:**
- Blurry reconstructions due to MSE loss
- Restricted posterior expressiveness (Gaussian assumption)
- Trade-off between reconstruction quality and latent regularization
- Single-shot generation limits detail refinement

**Why DDPMs:**
- Iterative refinement over many timesteps produces sharper images
- No encoder bottleneck during generation
- Better mode coverage and sample diversity
- State-of-the-art quality on complex datasets
- Flexible conditioning for text-to-image and other tasks

**Trade-off:** DDPMs are slower (100-1000 steps) but achieve superior quality compared to VAEs (1 step).

## Further Reading

- **Original VAE Paper**: Kingma & Welling (2013) - "Auto-Encoding Variational Bayes"
- **β-VAE**: Higgins et al. (2017) - Learning disentangled representations
- **Connection to Diffusion**: Sohl-Dickstein et al. (2015), Ho et al. (2020)

---

**Note:** This implementation uses a convolutional architecture optimized for image data. For other data types (text, audio), different architectures are needed.
