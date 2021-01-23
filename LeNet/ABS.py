#模型采用的是VGG16norm，每一层后面都有一个batchnorm操作，对于全连接层只有两层；
#可能是因为查看的这两个算法都只对卷积层剪枝，所以全连接层不重要可以改变
#先这样做，等会改回来
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy




def DRIVE(filter_list, num_t):
    abs_filter = [np.sum(np.abs(p)) for p in filter_list]
    small_rank = np.argsort(abs_filter)

    return small_rank[: num_t]

#conv1 1*10*5*5 输入通道是1，输出通道是10
def absolute_weight_sum_conv(conv_name, para_data, res, num_t):
    
    #if 'weight' in conv_name:
    p = copy.deepcopy(para_data)
    p = p.squeeze()
    filter_list = [p[i].squeeze().data.numpy() for i in range(p.size(0))]

    res[conv_name[ : 5]] = DRIVE(filter_list, num_t)

def absolute_weight_sum_fc(fc_name, para_data, res, num_t):
    
    #if 'weight' in conv_name:
    p = copy.deepcopy(para_data)
    p = p.squeeze()
    filter_list = [p[i].squeeze().data.numpy() for i in range(p.size(0))]

    res[fc_name[ : 3]] = DRIVE(filter_list, num_t)
    
def get_code(compress_rate, model_original, args, train_loader, test_loader, val_loader):
    model = copy.deepcopy(model_original)
    model.cpu()
    res_ABS = {}

    for name, para in model.named_parameters():
        if para.requires_grad:
            if 'weight' in name:
                num_t = int(para.data.shape[0] * compress_rate)
                if 'conv' in name:
                    absolute_weight_sum_conv(name, para, res_ABS, num_t)

     
    
    return res_ABS



