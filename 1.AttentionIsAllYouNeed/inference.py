import torch
from model import LanguageModel
from model import BasicTokenizer
import json



def main():
    experiment_id = "dd7be62c-55cc-4404-999f-41bcc93a6616"

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

    model.load_state_dict(torch.load(f"experiments/{experiment_id}/model_5.pth"))

    tokenizer = BasicTokenizer(path=f"experiments/{experiment_id}/tokenizer.json")

    text = "Mi mundo"

    tokens = tokenizer.tokenize(text)

    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

    text = ""
    for final_idx in model.generate(input_ids,max_new_tokens=2000):
        print(f"final_idx.shape: {final_idx.shape}")
        last_token = final_idx[0][-1]
        print(f"last_token: {last_token}")
        last_token = tokenizer.detokenize([str(last_token.item())])
        text += last_token
        print(text)

if __name__ == "__main__":
    main()