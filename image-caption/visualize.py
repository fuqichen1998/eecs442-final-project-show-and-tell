import torch
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
num_imgs = 50
check_point_name = "best_checkpoint_flickr8k.pth.tar"
dictionary_json_path = os.path.join("./preprocess_out", 'DICTIONARY_WORDS_' + dataset + '.json')

## Load checkpoint ##
encoder, decoder, decoder_opt, last_epoch, best_bleu_score = load_checkpoint(check_point_name)
word_dict = None
with open(dictionary_json_path, 'r') as file:
    word_dict = json.load(file)
dict_len = len(word_dict)

def search_caption(img, use_beam_search=False, beam_size=5):
    return []


def beam_search():
    return []

def decode_caption():
    return []

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'TEST',
                                                             transform=transforms.Compose([normalize])),
                                               batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    encoded_cap = search_caption(test_loader)
    decoded_cap = decode_caption()

    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(test_loader)):
        if i % 5 != 0:
            continue
        if i / 5 >= num_imgs:
            break
        # print(len(image), len(caps), len(caplens))
        print("Image ", image)
        print("True caption for ")
        print("Decoded caption for image ", " ", " is:\n", decoded_cap)
