import random
import torch
from torch import nn

# augmentation utils copy from BYOL
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def SimSiamMLP(dim, projection_size, hidden_size=4096, dropout=0.3):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )


class Classifier(nn.Module):
    def __init__(self, encoder, in_features, n_class, dropout=0.3, hidden_size=4096) -> None:
        super().__init__()
        self.mlp = SimSiamMLP(in_features, n_class, hidden_size=hidden_size, dropout=dropout)
        self.encoder = encoder

    def forward(self, img):
        embed = self.encoder(img)
        logits = self.mlp(embed)
        return logits
        
        # a = []
        # x = embed
        # for name, layer in self.mlp._modules.items():
        #     x = layer(x)
        #     a.append(x)
        # return x, embed, a[-6]