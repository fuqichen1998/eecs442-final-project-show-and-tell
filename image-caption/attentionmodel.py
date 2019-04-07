import torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def _init__(self, attention_size, encoder_size, decoder_size):
        super(Attention, self).__init__()
        #attention_dim = 512  
        #decoder_dim = 512
        #encoder_dim = 512 (2048 in original setting)
        self.attention = nn.Linear(attention_size, 1)
        self.encoder = nn.Linear(encoder_size, attention_size)
        self.decoder = nn.Linear(decoder_size, attention_size)
        self.soft = nn.Softmax(1)
    def forward(self, einput, dinput):
        x1 = self.encoder(einput)
        x2 = self.decoder(dinput).unsqueeze(1)
        out = nn.ReLU(x1+x2)
        out = self.attention(out).squeeze(2)
        out = self.soft(out)
        out1 = out.unsqueeze(2)
        weights = (einput*out1).sum(1)

        return out, weights


        