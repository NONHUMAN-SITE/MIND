import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicTokenizer:

    def __init__(self,text:str):
        self.string2int = {ch:i for i,ch in enumerate(text)}
        self.int2string = {i:ch for i,ch in enumerate(text)}

    def tokenize(self,text:str):
        return [self.string2int[ch] for ch in text]
    
    def detokenize(self,tokens:list[int]):
        return "".join([self.int2string[i] for i in tokens])
    
    def __len__(self):
        return len(self.string2int)
    

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self,
                 n_embd:int,
                 head_size:int,
                 dropout:float,
                 context_length:int):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Notas:
        * No importa el tamaño de la entrada, manejamos los tamaños adecuando las entradas y salidas con T.
        * Usamos wei.masked_fill para evitar que los tokens no puedan ver los tokens futuros.
        * wei solo es una forma más eficiente de implementar este masking.
        '''
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self,
                 n_embd:int,
                 num_heads:int,
                 head_size:int,
                 dropout:float,
                 context_length:int):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd,head_size,dropout,context_length) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self,
                 n_embd:int,
                 dropout:float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,
                 n_embd:int,
                 n_head:int,
                 dropout:float,
                 context_length:int):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd,n_head,head_size,dropout,context_length)
        self.ffwd = FeedFoward(n_embd,dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 context_length:int,
                 n_embd:int,
                 n_head:int,
                 n_layer:int,
                 dropout:float,
                 device:torch.device):
        
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(context_length,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head,dropout,context_length) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def positional_encoding(self,seq_len:int, d_model:int):
        """
        Genera el encoding posicional para un modelo de transformador.

        Args:
            seq_len (int): Longitud de la secuencia (número de posiciones).
            d_model (int): Dimensión del modelo (tamaño de los embeddings).

        Returns:
            torch.Tensor: Tensor de tamaño (seq_len, d_model) con los encodings posicionales.
        """
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self,idx,targets=None):
        B,T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.positional_encoding(T,self.n_embd).to(self.device)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # If we are not training, we don't need to compute the loss
        if targets is None:
            loss = None
            return logits,loss
        else:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
            return logits,loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 256
    context_length = 64
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.2

    model = LanguageModel(vocab_size,context_length,n_embd,n_head,n_layer,dropout,device)
    print(model)