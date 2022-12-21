import torch
import torch.nn as nn
from nets.vgg import VGG16
import torch.nn.functional as F

    
def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width)*get_output_length(height) 
    
class VGG_embedding(nn.Module):
    def __init__(self, input_shape, way,shot,query,pretrained=False):
        super(VGG_embedding, self).__init__()
        self.vgg = VGG16(pretrained, input_shape[-1])
        del self.vgg.avgpool
        del self.vgg.classifier

        self.way = way
        self.shot = shot
        self.query = query
        
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fc1 = nn.Linear(flat_shape,1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x,mode='support'):
        x = self.vgg.features(x)
        x = x.view(x.size()[0], -1)
        if mode == 'support':
            x = x.view(self.way, self.shot, -1).mean(1)
        elif mode == 'test_one':
            x = x
        else:
            x = x.view(self.way*self.query, -1)
        return x