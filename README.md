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
The encoder needs to extract image features of various sizes and encodes them into vector space which can be fed to RNN in a later stage. VGG-16 and ResNet is commonly recommened as image encoders. We chose to modify the pre-trained VGG-16 model provided by PyTorch library.

In this task, CNN is used to encode features instead of classify images. As a result, we removed the fully connected layers and the max pool layers at the end of the network. Under this new construction, the input image matrix has dimension N x 3 x 256 x 256, and the output has dimension N x 14 x 14 x 512. Furthermore, in order to support input images with various sizes, we added an adaptive 2d layer to our CNN architecture.

In our image captioning architecture, we disabled gradient to reduce computational costs. With fine-tuning, we might obtain a better overall performance.


## RNN (LSTM) Network
The decoder needs to generate image captions word by word using a Recurrent Neural Network - LSTMs which is able to sequentially generate words. The input for the decoder is the encoded image feature vectors from CNN and the encoded image captions produced in data preprocessing stage. 

The decoder consists of an attention module designed and implemented by ourselves, an LSTM cell module and four fully connected layers provided by PyTorch library for the initialization of the states of LSTMcell and word dictionary.

When receiving the encoded images and captions,  we first sort the encoded images and captions by encoded key length of images in descending order. We intend to only process the encoded images which have caption lengths greater than or equal to the number of iteration to increase efficiency and reduce training time.

In each iteration of LSTM network, we first put the historical state of LSTM and the encoded images into the Attention module to get the attention-masked images which have a specific highlighted area. Then we concatenate the embedded captions of all previous words and the attentioned images and feed them to the LSTM to get the next state of LSTM. Then, fully connected layers can predict the probabilities of current word embedding based on the current state and append it to the word embedding prediction matrix.

## Loss Function
We use Cross Entropy Loss

## Evaluation
Here we chose 4-gram BLEU score(BLEU-4) as our primary evaluation metric. The next step is to also investigate other metrics (such as METEOR) mentioned in relevant papers

## Result
See final report pdf
