import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from processor_pdf import merge_pdfs_text
from model import BasicTokenizer,LanguageModel
from dataset import GPTDataset
from logger import logger
import numpy as np
import uuid



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str, help="Folder with pdfs",required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size",default=64)
    parser.add_argument("--epochs", type=int, help="Number of epochs",default=30)
    parser.add_argument("--learning_rate", type=float, help="Learning rate",default=0.001)
    parser.add_argument("--save_every", type=int, help="Save every n epochs",default=10)
    return parser.parse_args()


def main():
    # Process arguments
    args = get_args()
    os.makedirs("models",exist_ok=True)
    os.makedirs("experiments",exist_ok=True)
    experiment_id = str(uuid.uuid4())
    os.makedirs(f"experiments/{experiment_id}",exist_ok=True)

    # Merge pdfs
    logger.log(f"Merging pdfs: {args.dataset}")
    pdfs_paths = [os.path.join(args.dataset,file) for file in os.listdir(args.dataset)]
    text = merge_pdfs_text(pdfs_paths=pdfs_paths)
    logger.log(f"Total characters: {len(text)}")

    summary_writer = SummaryWriter(log_dir="runs/")

    context_length = 64

    # Create tokenizer
    logger.log(f"Tokenizing text")
    tokenizer = BasicTokenizer(text)
    tokenizer.save_tokenizer(f"experiments/{experiment_id}/tokenizer.json")

    # Create datasets
    train_dataset = GPTDataset(text,"train",0.8,context_length=context_length,tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

    test_dataset = GPTDataset(text,"test",0.8,context_length=context_length,tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

    # Create model
    vocab_size = len(sorted(list(set(text))))
    logger.log(f"Vocab size: {vocab_size}")
    
    logger.log(f"Creating model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageModel(vocab_size=vocab_size,
                          context_length=context_length,
                          n_embd=256,
                          n_head=4,
                          n_layer=4,
                          dropout=0.2,
                          device=device)
    model.save_parameters(f"experiments/{experiment_id}/model_parameters.json")
    model.to(device)
    
    logger.log(f"Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)

    logger.log(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    logger.log(f"Training...")
    step = 0
    loss_array = []
    for epoch in range(args.epochs):
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
            loss_array.append(loss.item())
            summary_writer.add_scalar("Loss/train",loss.item(),step)
            step += 1
        
        if (epoch+1) % args.save_every == 0:
            torch.save(model.state_dict(),f"experiments/{experiment_id}/model_{epoch+1}.pth")

    np.savetxt(f"experiments/{experiment_id}/loss.txt",loss_array)
    summary_writer.close()

    logger.log(f"Training finished")

if __name__ == "__main__":
    main()

