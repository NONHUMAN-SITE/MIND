# Attention Is All You Need

This section contains everything related to the seminal paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). You can find the complete blog and additional resources at:  
[www.wiki.nonhuman.site/projects/MIND/1.1AttentionIsAllYouNeed](https://www.wiki.nonhuman.site/projects/MIND/1.1AttentionIsAllYouNeed)

---

## Installation

To set up the environment, please follow the installation steps provided in the original repository's main README.

---

## Training Instructions

1. **Set Permissions**  
   Make the `run.sh` script executable:  
   ```bash
   chmod +x ./run.sh
   ```

2. **Configure File Paths**  
   Inside the `run.sh` file, specify the path where the PDF files are located. Ensure that the text within the PDF files is selectable and not just images.

3. **Minimum Data Requirements**  
   For optimal results, ensure your dataset contains at least **1 million characters**.

4. **Hyperparameter Configuration**  
   - Hyperparameters for the model can be modified in `train.py`.  
   - Training-specific parameters can be adjusted in `run.sh`.

5. **GPU/CPU Usage**  
   - If a GPU is available, the script will utilize it automatically.  
   - If no GPU is available, the CPU will be used (not recommended for large datasets).

---

## Visualizing Training with TensorBoard

To better visualize the training process:  

1. Run the following command after executing the training script:  
   ```bash
   tensorboard --logdir=runs
   ```

2. Open the link provided in the terminal to access TensorBoard in your browser.  

---

**Note**: This section is part of the broader "MIND" project. For more details and resources, visit the main repository: [MIND](https://github.com/NONHUMAN-SITE/MIND).

