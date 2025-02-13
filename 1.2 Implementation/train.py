import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import uuid
import yaml

from src.utils import merge_pdfs_text,setup_experiment
from src.model import BasicTokenizer,LanguageModel
from src.dataset import GPTDataset
from src.logger import logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Folder with pdfs", required=True)
    parser.add_argument("--config", type=str, help="Path to config file", default="config.yaml")
    return parser.parse_args()

def main():
    # Process arguments
    args = get_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    experiment_id = str(uuid.uuid4())

    experiment_dir,models_dir,tensorboard_dir = setup_experiment(config,experiment_id)
    
    # Save config in experiment directory
    config_path = f"{experiment_dir}/config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.log(f"Experiment ID: {experiment_id}")
    logger.log(f"Config saved to: {config_path}")
    logger.log(f"Tensorboard logs will be saved to: {tensorboard_dir}")

    # Merge pdfs
    logger.log(f"Merging pdfs: {args.dataset}")
    pdfs_paths = [os.path.join(args.dataset,file) for file in os.listdir(args.dataset)]
    text = merge_pdfs_text(pdfs_paths=pdfs_paths)
    logger.log(f"Total characters: {len(text)}")

    # Initialize tensorboard writer with experiment directory
    summary_writer = SummaryWriter(log_dir=tensorboard_dir)

    # Create tokenizer
    logger.log(f"Tokenizing text")
    tokenizer = BasicTokenizer(text)
    tokenizer.save_tokenizer(f"experiments/{experiment_id}/tokenizer.json")

    # Create datasets
    train_dataset = GPTDataset(text,
                               context_length=config['training']['context_length'],
                               tokenizer=tokenizer)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config['training']['batch_size'],
                                               shuffle=True)

    # Create model
    vocab_size = tokenizer.get_vocab_size()
    logger.log(f"Vocab size: {vocab_size}")
    
    logger.log(f"Creating model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageModel(
        vocab_size=vocab_size,
        context_length=config['training']['context_length'],
        n_embd=config['model']['n_embd'],
        n_head=config['model']['n_head'],
        n_layer=config['model']['n_layer'],
        dropout=config['model']['dropout'],
        device=device
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    num_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Total parameters: {num_params} ({(num_params/1e6):.2f}M)")

    logger.log(f"Training...")
    step = 0
    for epoch in range(config['training']['epochs']):
        loss_epoch = 0
        for batch in tqdm(train_loader,total=len(train_loader),desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids,labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            output,loss = model(input_ids,labels)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            summary_writer.add_scalar("Loss/train",loss.item(),step)
            step += 1
        
        if (epoch+1) % config['training']['save_every'] == 0:
            torch.save(model.state_dict(), f"{models_dir}/model_{epoch+1}.pth")
        
        torch.save(model.state_dict(), f"{models_dir}/last.pth")

    summary_writer.close()

    logger.log(f"Training finished. All data saved in {experiment_dir}")

if __name__ == "__main__":
    main()

