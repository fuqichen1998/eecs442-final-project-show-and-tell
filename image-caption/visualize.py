import torch
import matplotlib as plt
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from dataset import *
from helper import *
import imageio
from tqdm import tqdm
import torch.nn.functional as Func

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


def search_caption(img_tensor, use_beam_search=False, beam_size=5):
    encode = encoder(img_tensor)  # 1*14*14*512
    encode = encode.view(1, encode.size(1) * encode.size(2), 512)  # encode channel = 512 result = 1*(14*14)*512
    out = encode.expand(beam_size, encode.size(1), 512)
    return


def beam_search():
    return None


def decode_caption():
    return None


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
test_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'TEST',
                                                        transform=transforms.Compose([normalize])),
                                          batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
for i, (image, caps, caplens, allcaps) in enumerate(test_loader):
    if i % 5 != 0:
        continue
    if i / 5 >= num_imgs:
        break
    # already resized and permuted in dataset
    encoded_cap = search_caption(image.to(device))
    decoded_cap = decode_caption()
    imageio.imwrite(os.path.join(output_path, str(i // num_imgs) + ".jpg"), image.numpy()[0].transpose(1, 2, 0))

# Write all decoded caps into a text file

