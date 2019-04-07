import torch
import torchvision
from torch import nn
from attentionmodel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMs(nn.Module):
    def __init__(self, encoder_dim, attention_dim, embed_dim, decoder_dim, dic_size, dropout=0.5):
        super(LSTMs, self).__init__()

        # dimensions
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.dic_size = dic_size

        #attention
        self.attention = Attention(attention_dim, encoder_dim, decoder_dim)

        # modules
        self.embedding = nn.Embedding(dic_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.h = nn.Linear(encoder_dim, decoder_dim)
        self.c = nn.Linear(encoder_dim, decoder_dim)
        self.fc_sig = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc_dic = nn.Linear(decoder_dim, dic_size)
        
        # weight initialization
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                nn.init.constant_(m.bias, 0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, encoder_cap, cap_len):

        # flatten image to size (batch_size, sumofpixcel, encoder_dim)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        dic_size = self.dic_size

        # sort data in length of caption (useful in the for loop of LSTMs to reduce time)
        sorted_cap_len, sorted_index = torch.sort(cap_len.squeeze(1), dim=0, descending=True)
        encoder_out = encoder_out[sorted_index]
        encoder_cap = encoder_cap[sorted_index]

        # embedding
        cap_embedding = self.embedding(encoder_cap)

        # initializa LSTM Cell
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.h(mean_encoder_out)
        c = self.c(mean_encoder_out)

        # leave the last work <end>
        decoder_len = (sorted_cap_len - 1).tolist()
        # print(decoder_len)
        max_length = max(decoder_len)

        # initialize the output predictions and alpha
        # print(batch_size)
        # print(max_length)
        # print(dic_size)

        predictions = torch.zeros(batch_size, max_length, dic_size).to(device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(device)

        # loop over the max length of caption
        for i in range(max_length):
            # the sub batch have length of caption greater than i
            # should be put into i^th LSTM
            subatch_index = sum([l > i for l in decoder_len])

            # attention area
            attention_area, alpha = self.attention(encoder_out[:subatch_index], h[:subatch_index])
            mask = self.sigmoid(self.fc_sig(h[:subatch_index]))
            attentioned_out = mask * attention_area

            # run LSTM
            h, c = self.lstm(torch.cat([cap_embedding[:subatch_index, i, :], attentioned_out], dim=1)
                    , (h[:subatch_index], c[:subatch_index]))
            preds = self.fc_dic(self.dropout(h))

            #append result
            predictions[:subatch_index, i, :] = preds
            alphas[:subatch_index, i, :] = alpha

        return predictions, alphas

        


    
