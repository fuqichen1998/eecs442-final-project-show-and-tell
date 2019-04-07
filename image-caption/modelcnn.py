import torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CNN(nn.Module):
    def __init__(self, encodingsize = 14):
        super(CNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        layerlist = list(vgg16.children())
        self.vgg16 = nn.Sequential(*layerlist[:-4]) #removed all pool including maxpool
        self.pool_reshape = nn.AdaptiveAvgPool2d((encodingsize,encodingsize))
        for parameter in self.vgg16.parameters():
            parameter.requires_grad = False #TODO:check correctness/fine tune
    def forward(self, input):
        #input dimension: batchsize*3*224*224
        x = self.vgg16(input) 
        #input dimension: batchsize*512*14*14 (inputH/16)
        x = self.pool_reshape(x)
        #input dimension: batchsize*512*14*14 (inputH/16)
        x = x.permute(0, 2, 3, 1)
        #output dimension: batchsize*14*14*512 #TODO:check
        return x
