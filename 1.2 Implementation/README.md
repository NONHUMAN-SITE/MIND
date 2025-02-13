# Attention Is All You Need Implementation

This section contains the implementation of the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

All model and training parameters can be configured in `config.yaml`. The main parameters include:

```yaml
training:
  batch_size: 64
  epochs: 30
  learning_rate: 0.0001
  save_every: 10
  context_length: 64

model:
  n_embd: 256
  n_head: 4
  n_layer: 4
  dropout: 0.2
```

## Training

1. **Prepare Your Data**
   - Create a folder containing your PDF files
   - Ensure the text within PDFs is selectable (not images)
   - Recommended: at least 1 million characters for optimal results

2. **Start Training**
```bash
python train.py --dataset path_to_data
```
where `path_to_data` is the folder containing your PDF files.

The training will:
- Create a unique experiment directory under `experiments/`
- Save model checkpoints, configuration, and tensorboard logs
- Display training progress in the terminal

## Inference

To generate text using a trained model:

1. Locate your experiment directory (it will be shown during training)
2. Modify the `test.py` file to use the correct experiment directory
3. Run:
```bash
python test.py
```
4. When prompted, enter your initial text
5. The model will generate text continuing from your input

## Monitoring Training

To visualize training metrics:

1. Navigate to your experiment directory:
```bash
cd experiments/your_experiment_id
```

2. Start TensorBoard:
```bash
tensorboard --logdir=tensorboard
```

3. Open the provided URL in your browser (typically `http://localhost:6006`)

## Directory Structure

After training, your experiment directory will contain:
```
experiments/
└── your_experiment_id/
    ├── config.yaml           # Configuration used for training
    ├── models/              
    │   └── last.pth         # Model weights
    ├── tokenizer.json       # Tokenizer configuration
    └── tensorboard/         # Training logs
```

---

For more details and resources, visit: [www.nonhuman.site/research/wiki/MIND/1.2](https://www.nonhuman.site/research/MIND/1.2)

