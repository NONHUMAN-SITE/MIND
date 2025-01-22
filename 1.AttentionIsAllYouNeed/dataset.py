import torch
from torch.utils.data import Dataset
from model import BasicTokenizer


class GPTDataset(Dataset):

    def __init__(self,
                 text:str,
                 type:str,
                 split:float,
                 context_length:int,    
                 tokenizer):
        '''
        text: str, This is the text to be tokenized
        type: str, This is the type of the dataset, it can be "train" or "test"
        split: float, This is the split of the dataset, it can be a float between 0 and 1
        context_length: int, This is the context length of the dataset
        tokenizer: This is the tokenizer to be used
        '''
        
        assert type in ["train","test"]
        assert 0 <= split <= 1

        if type == "train":
            self.data = torch.tensor(tokenizer.tokenize(text[:int(len(text)*split)]))
        else:
            self.data = torch.tensor(tokenizer.tokenize(text[int(len(text)*split):]))
        
        self.tokenizer = tokenizer
        self.context_length = context_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        idx = min(idx,len(self.data)-self.context_length-1) # This is to ensure that the index is within the bounds of the dataset
        x = torch.stack([self.data[idx:idx+self.context_length]])
        y = torch.stack([self.data[idx+1:idx+self.context_length+1]])
        return x.squeeze(),y.squeeze()