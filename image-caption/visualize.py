import torch
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from dataset import *
from helper import *
from tqdm import tqdm
import torch.nn.functional as Func

check_point_name = "best_checkpoint_flickr8k.pth.tar"
dictionary_json_path = ""

encoder, decoder, decoder_opt, last_epoch, best_bleu_score = load_checkpoint(check_point_name)

def visualize_caption(use_beam_search=False, beam_size=5):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(CustomDataset("./preprocess_out", "flickr8k", 'TEST',
                                                             transform=transforms.Compose([normalize])),
                                               batch_size=48, shuffle=True, num_workers=1, pin_memory=True)

    # Load dictionary
    word_dict = None
    with open(dictionary_json_path, 'r') as file:
        word_dict = json.load(file)
    rev_word_map = {v: k for k, v in word_dict.items()}
    vocab_size = len(word_dict)

    return

if __name__ == '__main__':
    print("bleu-4 score is ", visualize_caption())
