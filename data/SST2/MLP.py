import torch
import torch.nn as nn
import torch.nn.functional as F


# build a feedforward neural network
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 50
        self.vocab_size = 14762 
        self.embedding_dim = 300
        self.target_dim = 2
        
        self.embedding = nn.Embedding(
            self.vocab_size, 
            self.embedding_dim
        )
        self.lin = nn.Linear(
            self.input_size * self.embedding_dim, 
            self.target_dim
        )
        
    def forward(self, x, x_emb=None):
        if x_emb is None:
            x_emb = self.embedding(x)
        features = x_emb.view(x.size()[0], -1)
        features = F.relu(features)
        features = self.lin(features)
        return features
