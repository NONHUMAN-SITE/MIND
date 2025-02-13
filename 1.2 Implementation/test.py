import torch
from model import LanguageModel
from model import BasicTokenizer
import json


def main():
    experiment_id = "a971053c-dc7b-4633-a224-1c6fac23107f"

    parameters = json.load(open(f"experiments/{experiment_id}/model_parameters.json","r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LanguageModel(vocab_size=parameters["vocab_size"],
                          context_length=parameters["context_length"],
                          n_embd=parameters["n_embd"],
                          n_head=parameters["n_head"],
                          n_layer=parameters["n_layer"],
                          dropout=parameters["dropout"],
                          device=device)
    
    model.to(device)

    model.load_state_dict(torch.load(f"experiments/{experiment_id}/model_23.pth", weights_only=True))

    tokenizer = BasicTokenizer(path=f"experiments/{experiment_id}/tokenizer.json")

    text = "Mi mundo"

    tokens = tokenizer.tokenize(text)

    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

    text = "."
    print(f"Texto inicial: {text}")
    for last_token in model.generate(input_ids,max_new_tokens=4000):
        last_token = tokenizer.detokenize([str(last_token)])
        print(last_token,end="")

if __name__ == "__main__":
    main()