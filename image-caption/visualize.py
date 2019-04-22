import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from dataset import *
from torch import nn
from helper import *
import numpy as np
import random
import imageio
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import torch.nn.functional as F

## Parameters ##
output_path = "./test_caption_out"
dataset = "flickr8k"
num_imgs = 10
check_point_name = "best_checkpoint_flickr8k.pth.tar"
dictionary_json_path = os.path.join("./preprocess_out", 'DICTIONARY_WORDS_' + dataset + '.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load checkpoint ##
encoder, decoder, decoder_opt, last_epoch, best_bleu_score = load_checkpoint(check_point_name)

word_dict = None
with open(dictionary_json_path, 'r') as file:
    word_dict = json.load(file)
dict_len = len(word_dict)

reversed_word_dict = {}
for k, v in word_dict.items():
    reversed_word_dict[v] = k


def search_caption(img_tensor, use_beam_search=False, beam_size=5):
    encode = encoder(img_tensor)  # 1*14*14*512
    encode = encode.view(1, encode.size(1) * encode.size(2), 512)  # encode channel = 512 result = 1*(14*14)*512
    out = encode.expand(beam_size, encode.size(1), 512)
    return


def validate(normalize):
    alpha_c = 1.  # regularization parameter for 'doubly stochastic attention'
    val_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'VAL',
                                                           transform=transforms.Compose([normalize])),
                                             batch_size=48, shuffle=True, num_workers=1, pin_memory=True)
    # validate and return score
    val_loss_all = 0
    references = []
    hypotheses = []
    #######################################
    # TODO: check if with torch.no_grad(): necessary
    decoder.eval()
    with torch.no_grad():
        for i, (img, caption, cap_len, all_captions) in enumerate(val_loader):
            # use GPU if possible
            img = img.to(device)
            caption = caption.to(device)
            cap_len = cap_len.to(device)

            # forward
            encoded = encoder(img)
            preds, sorted_caps, decoded_len, alphas, sorted_index = decoder(
                encoded, caption, cap_len)

            # ignore the begin word
            trues = sorted_caps[:, 1:]
            preds2 = preds.clone()
            # pack and pad
            preds, _ = pack_padded_sequence(preds, decoded_len, batch_first=True)
            trues, _ = pack_padded_sequence(trues, decoded_len, batch_first=True)

            # calculate loss
            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(preds, trues)
            loss += alpha_c * (1. - alphas.sum(dim=1) ** 2).mean()
            val_loss_all += loss

            # TODO: print performance
            all_captions = all_captions[sorted_index]
            for j in range(all_captions.shape[0]):
                img_caps = all_captions[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_dict['<start>'], word_dict['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)
            _, predmax = torch.max(preds2, dim=2)
            predmax = predmax.tolist()
            temp_preds = list()
            for j, p in enumerate(predmax):
                temp_preds.append(
                    predmax[j][:decoded_len[j]])  # remove pads
            predmax = temp_preds
            hypotheses.extend(predmax)
            assert len(references) == len(hypotheses)

        print("Validation Loss All: ", val_loss_all)
        bleu4 = corpus_bleu(references, hypotheses)
        print("bleu4 score: ", bleu4)


def decode_caption(encoded_words, img):
    decoded_sentence = []
    for encoded_word in encoded_words:
        decoded_sentence.append(reversed_word_dict[encoded_word])
    plt.text(0, 1, "".join(decoded_sentence))
    plt.imshow(img)
    # plt.axis('off')
    plt.imsave(output_path, )
    # plt.show()
    return decoded_sentence

def visualize_cap_with_attention(encoded_words, img, alphas):
    decoded_sentence = decode_caption(encoded_words, img)
    for idx, seq in enumerate(decoded_sentence):
        plt.text(0, 1, decoded_sentence[idx])
        plt.imshow(img)
        alpha_vector = alphas[idx, :]
        alpha_value = 0.5 if idx == 0 else 0
        # https://matplotlib.org/gallery/color/colormap_reference.html
        plt.set_cmap(cm.Reds)
        plt.imshow(alpha_vector, alpha=alpha_value)
        # plt.axis('off')
        plt.show()


if __name__ == '__main__':
    random.seed(442)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # validate(normalize)
    test_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'TEST',
                                                            transform=transforms.Compose([normalize])),
                                              batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    for i, (image, caps, caplens, allcaps) in enumerate(test_loader):
        if i % 5 != 0:
            continue
        if i / 5 >= num_imgs:
            break
        # already resized and permuted in dataset
        # encoded_cap = search_caption(image.to(device))
        encoded_cap = (np.random.rand(30) * 1000).astype(int)
        # dimension of image to 1 x W x H x 3
        img_numpy = image.numpy()[0].transpose(1, 2, 0)
        img_numpy = img_numpy.astype(np.uint8)
        alphas = np.random.rand(30, img_numpy.shape[1], img_numpy.shape[2])
        visualize_cap_with_attention(encoded_cap, img_numpy, alphas=alphas)
        # decoded_sentence = decode_caption(encoded_cap, image.numpy()[0].transpose(1, 2, 0))
        # imageio.imwrite(os.path.join(output_path, str(i // num_imgs) + ".jpg"), image.numpy()[0].transpose(1, 2, 0))
    # Write all decoded caps into a text file

