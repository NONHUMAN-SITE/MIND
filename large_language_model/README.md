# Large Language Model Educational Content

This directory contains educational materials for understanding and experimenting with Large Language Models (LLMs), specifically using Google's Gemma models.

> This section was created by [Jorge Muñoz Laredo](https://jorgemunozl.github.io) active member of the [NONHUMAN](https://nonhuman.site) organization.

## Contents

### 📓 Main Notebooks

#### `large_language_model.ipynb`
**Gemma Under the Hood: A Deep Dive into Model Architecture**

This comprehensive notebook explores the internal workings of Gemma models (2B vs 7B parameters):

**Topics covered:**
1. **Encoding and Model Architecture**
   - How Gemma's tokenizer processes text
   - Inspecting model structure (embeddings, transformer blocks, attention projections)

2. **Forward Pass Mechanics**
   - Understanding how input tokens flow through the model
   - Converting logits to probabilities
   - Generating predictions

3. **Comparative Experiments**
   - Parameter count differences between 2B and 7B models
   - VRAM usage and inference speed comparisons
   - Output quality analysis
   - Attention weight visualization

**Prerequisites:**
- Python with PyTorch
- HuggingFace Transformers library
- CUDA-capable GPU (recommended)

**Getting Started:**
```bash
pip install transformers huggingface_hub torch
```

#### `gemma_Inference_samples.ipynb`
**Practical Inference with Gemma 2B**

A hands-on notebook demonstrating inference using Gemma's 2B parameter variant on various prompt types.

**Use this notebook to:**
- Learn practical inference techniques
- Test different prompting strategies
- Understand model behavior on diverse tasks

### 🧪 Test Prompts

#### `prompts.py`
A curated collection of 15 diverse prompts designed to test different aspects of LLM capabilities:

**Categories:**
- **Reasoning & Logic**: Theory of mind puzzles, counting problems, math calculations
- **Constraint Following**: Writing with letter restrictions, format constraints
- **Knowledge Tasks**: Scientific explanations (quantum mechanics, mathematics)
- **Creative & Stylistic**: Humor, irony, audience adaptation
- **Edge Cases**: Safety testing, complex mathematical integrals

**Example usage:**
```python
from prompts import prompts

# Access individual prompts
theory_of_mind = prompts[0]  # Alice and Bob puzzle
constraint_writing = prompts[1]  # No letter 'e' challenge
math_problem = prompts[12]  # Arithmetic calculation
calculus = prompts[13]  # Integration problem
```

### 🖼️ Visualizations

- `attn_test.png`: Attention weight visualization
- `v1.png`, `v2.png`: Additional model analysis visualizations

## Recommended Learning Path

1. **Start with `large_language_model.ipynb`**
   - Work through sections sequentially
   - Experiment with both 2B and 7B models if resources allow
   - Pay attention to attention visualizations

2. **Explore `gemma_Inference_samples.ipynb`**
   - Run inference examples
   - Modify parameters to see their effects

3. **Experiment with `prompts.py`**
   - Import prompts into either notebook
   - Test model responses across different prompt types
   - Compare outputs between model sizes

## System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- GPU with 6GB+ VRAM (for 2B model)

**Recommended:**
- Python 3.10+
- 16GB+ RAM
- GPU with 16GB+ VRAM (for 7B model)

**For CPU-only inference:**
- Expect significantly slower performance
- 2B model is more practical than 7B

## GPU Configuration

Both notebooks include GPU selection code for multi-GPU systems:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
```

Modify the index to select different GPUs or use multiple GPUs.

## Key Concepts You'll Learn

- **Tokenization**: How text is converted to model inputs
- **Embeddings**: Vector representations of tokens
- **Attention Mechanisms**: How models focus on relevant information
- **Transformer Architecture**: Layer-by-layer processing
- **Logits & Probabilities**: Converting model outputs to predictions
- **Model Scaling**: Trade-offs between model size, speed, and quality
- **Prompt Engineering**: Crafting effective inputs for different tasks

## Troubleshooting

**Out of Memory (OOM) errors:**
- Use the smaller 2B model
- Reduce batch size
- Enable gradient checkpointing
- Use quantization (8-bit or 4-bit)

**Slow inference:**
- Ensure you're using GPU, not CPU
- Check CUDA is properly installed: `torch.cuda.is_available()`
- Consider using smaller models for experimentation

**Model download issues:**
- Ensure you have HuggingFace Hub access
- Some models may require authentication: `huggingface-cli login`
- Check your internet connection

## Further Exploration

After completing these materials, consider:
- Fine-tuning Gemma on custom datasets
- Implementing custom attention visualizations
- Comparing with other model architectures (LLaMA, Mistral)
- Exploring quantization techniques for efficient deployment
- Building applications using the inference patterns learned

## Resources

- [Gemma Documentation](https://ai.google.dev/gemma)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**Note:** These materials are designed for educational purposes. Ensure you comply with model licenses and usage terms when deploying in production environments.
