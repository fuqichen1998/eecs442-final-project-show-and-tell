from lstms import *
import torch

if __name__ == "__main__":
    input = torch.randn((5, 32, 32, 64))

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    
    encoder_out = torch.rand((100, 32, 32, 512))
    encoder_cap = torch.rand((100, 20))
    encoder_cap = torch.randint((100, 1))
    decoder = LSTMs(encoder_dim=512, 
                attention_dim=attention_dim,
                embed_dim=emb_dim,
                decoder_dim=decoder_dim,
                dic_size=5,
                dropout=dropout)

    out = decoder