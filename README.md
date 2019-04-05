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

### Data Preprocessing
First if we use pretrained well-known CNN provided in torch vision, we need to preprocess out image
data into a specific input format that torch vision model accepts, referring to this page:
https://pytorch.org/docs/master/torchvision/models.html


## CNN Network
TODO


## RNN (LSTM) Network
TODO


## Evaluation
TODO


## Result
TODO
