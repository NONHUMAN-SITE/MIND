import torch
import argparse
import yaml
from src.model import LanguageModel, BasicTokenizer
from src.logger import logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to experiment directory")
    return parser.parse_args()

def main():
    args = get_args()
    experiment_dir = args.experiment_dir

    # Load config from experiment directory
    logger.log(f"Loading config from {experiment_dir}/config.yaml")
    with open(f"{experiment_dir}/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # Create model using parameters from config
    model = LanguageModel(
        vocab_size=config['model']['vocab_size'],
        context_length=config['training']['context_length'],
        n_embd=config['model']['n_embd'],
        n_head=config['model']['n_head'],
        n_layer=config['model']['n_layer'],
        dropout=config['model']['dropout'],
        device=device
    )
    model.to(device)

    # Load model weights
    model_path = f"{experiment_dir}/models/last.pth"
    logger.log(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load tokenizer
    tokenizer_path = f"{experiment_dir}/tokenizer.json"
    logger.log(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = BasicTokenizer(path=tokenizer_path)

    # Get input from user
    text = input("Put some text: ")
    logger.log(f"Input text: {text}")

    # Tokenize input
    tokens = tokenizer.tokenize(text)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

    # Generate text
    logger.log("Generating text...")
    print("\nGenerated text:", end=" ")
    print(text, end="")
    for last_token in model.generate(input_ids, max_new_tokens=4000):
        token_text = tokenizer.detokenize([str(last_token)])
        print(token_text, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    main()