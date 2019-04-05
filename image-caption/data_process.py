import numpy as np
import os
from tqdm import tqdm
import json
from scipy.misc import imread, imresize
from collections import Counter
import random
import h5py


def process_images_captions(dataset='coco', cap_json_path='./caption_datasets/dataset_coco.json', img_path='./img_train',
                            out_path='./preprocess_out', min_word_freq=5, max_cap_len=80, caps_per_img=5):
    """
    Process the image datasets and also the caption json file associated with that image datasets.
    The default dataset is coco, but also supports flickr 8k and flickr 30k
    Output Files:
    1. HDF5 file containing images for each train, val, test in an I, 3, 256, 256 tensor.
       Pixels are unsigned 8 bit integer in range [0, 255]
    2. JSON file with encoded captions
    3. JSON file with caption lengths
    4. JSON file with word embedding map

    :param dataset: Can be one of 'coco', 'flickr8k', 'flickr30k'
    :param cap_json_path: JSON file that preprocesses image caption labels, see readme file for details
    :param img_path: path that contain all training images. If using coco data set, make sure immg_path contains two
    sub folders "train2014/" and "val2014/" which will be the COCO data downloaded from COCO website
    :param out_path: ouput files will be written to this folder
    :param min_word_freq: The minimum word frequency to be considered in wordmap, if smaller than min_word_freq
    <unk> token will be used
    :param max_cap_len: only consider captions shorter than max caption length
    :param caps_per_img: Captions per image that will be compiled to output file
    :return: No return. All preprocessed files will be written to out_path
    """
    # TODO: TO implement this function
    return


if __name__== "__main__":
    process_images_captions()