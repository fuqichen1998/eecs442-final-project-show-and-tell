from lstms import *
import torch

if __name__ == "__main__":
    input = torch.randn((5, 32, 32, 64))
    # net = LSTMs(input.size(3), )