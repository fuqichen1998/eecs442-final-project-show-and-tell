# EECS442 Final Project Winter 2019
## Project Proposal
https://docs.google.com/document/d/1MhsKnpK4VmU3S7HugB5dzHH0nB5fJzRtQ00tXBrWKt0/edit?usp=sharing


## Resources
### Show, Attend and tell
* Paper: https://arxiv.org/abs/1502.03044
* Original Implementation by authors: https://github.com/kelvinxu/arctic-captions
* Another net implemented by Karpathy: https://github.com/karpathy/neuraltalk2

### image caption generation evaluation tool
This toolkit can evaluate caption generation results in terms of these metrics: BLEU, METEOR, ROUGE-L, and CIDEr  
https://github.com/tylin/coco-caption


## Data Processing

### Training Data Download
This project uses COCO 14' Dataset + preprocessed image captions which will be used as training and validation
* I did not includ COCO dataset into this git repo due to its size (13GB for train and 6GB for val).
Please download the COCO data set from here:   
2014 Train Images : http://images.cocodataset.org/zips/train2014.zip  
2014 Validation Images: http://images.cocodataset.org/zips/val2014.zip  

* This is where I get JSON file for preprocessed COCO caption annotations from Karpathy:  
https://cs.stanford.edu/people/karpathy/deepimagesent/  
Download it from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

Note: we also find Flickr 8k and Flickr 30k info in Karpathy's public available preprocessed captions, and will try 
training our network on Flickr 8k first since it is much more light weight.  
I find two unofficial sources of training images data:  
Flickr 8k: https://www.kaggle.com/srbhshinde/flickr8k-sau  
Flickr 30k: https://www.kaggle.com/hsankesara/flickr-image-dataset  

### Place data
Place the unzipped image data into the "/image-caption/img_train" foler.  
If the dataset is flickr 8k or flickr 30k, img_train folder should have a bunch of images;
If the dataset is COCO, img_train folder should have 2 folders, `train2014/` and `val2014/`

### Data Preprocessing
First if we use pretrained well-known CNN provided in torch vision, we need to preprocess out image
data into a specific input format that torch vision model accepts, referring to this page:
https://pytorch.org/docs/master/torchvision/models.html
  
Then we will also need to create a data loader which will be used to load each training/val/test data
and fed to pytorch CNN RNN.  
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


## CNN Network
TODO


## RNN (LSTM) Network
TODO


## Evaluation
TODO


## Result
TODO
