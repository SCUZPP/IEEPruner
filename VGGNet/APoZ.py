#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
import argument
import os
# In[2]:

#返回每一层的的feature map，先用老方法，再用新方法做

class FilterAPoZ:
    def __init__(self, model):
        self.model = model
        self.fc = dict()
        self.ff = dict()
        
    def forward(self, x):
        #conv
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            #ReLU
            if isinstance(module, torch.nn.modules.conv.Conv2d):
            #if isinstance(module, torch.nn.modules.ReLU):
                    
                cell_value = x.data.cpu().numpy()
                if name in self.fc:
                    self.fc['%s' % name] += np.array([np.sum(cell_value[:,j,:] <= 0) for j in range(x.shape[1])])
                else:
                    self.fc['%s' % name] = np.array([np.sum(cell_value[:,j,:] <= 0) for j in range(x.shape[1])])
                #print(len(self.fc)-1, np.sum(self.fc[-1]))#.data.cpu().numpy()))


        x = x.view(x.size(0), -1)
        
        #全连接层，不知道全连接层取linear怎么取
        for layer, (name, module) in enumerate(self.model.classifier._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.Linear):
                #print(name)
                cell_value = x.data.cpu().numpy()
                if name in self.ff:
                    self.ff['%s' % name] += np.array([np.sum(cell_value[:,j] <= 0) for j in range(x.shape[1])])
                else:
                    self.ff['%s' % name] = np.array([np.sum(cell_value[:,j] <= 0) for j in range(x.shape[1])])
        #for i in range(len(self.fc)):
        #    print(i, np.sum(self.fc[i]))#.data.cpu().numpy()))
        return x
    
    def get_conv_fp(self):
        return self.fc
    
    def get_fc_fp(self):
        return self.ff


#计算每层的输出feature map的average percentage of activation（APoZ），排序，取每层最小的n个神经元
#标记对应生成这个feature map的filters为可能被剪掉的filters


def Merge(dict1, dict2):
    return(dict1.update(dict2))


def APoZ(compress_rate, model, args, train_loader, test_loader, val_loader):
    model.eval()
    val_loss = 0
    val_acc = 0
    test_loss = 0
    
    #count_conv_fp = conv_fp_dict
    #count_fc_fp = fc_fp_dict
    
        
    #print(count_conv_fp)
    #print(conv_fp_dict)
    count = 0
    pruner = FilterAPoZ(model)
    
    for img, lable in val_loader:
        if args.cuda:
            #64*1*28*28
            img = img.cuda()
            #64
            lable = lable.cuda()
            
        img = Variable(img)
        #64
        lable = Variable(lable)
  
        #print('forward')
        out = pruner.forward(img)
        
        loss = F.cross_entropy(out, lable)
        test_loss = loss.item() # sum up batch loss
        pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(lable.data.view_as(pred)).cpu().sum()
        
        
            
        if args.cuda:
            #args.test_batch_size = 1
            test_loss /= img.size(0)
            num = img.size(0)
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, num,
                100. * correct / num)) 
        
        else:  
            num = img.size(0)
            test_loss /= num
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, num,
                100. * correct / num))    
        #break
        
    count_conv_fp = pruner.get_conv_fp()
    count_fc_fp = pruner.get_fc_fp()
                
    
    for (name, value) in count_conv_fp.items():
        #value = np.array(value)
        #print('name', name)
        #print('count_conv_fp[name]', count_conv_fp[name])
        num_t = int(compress_rate * value.shape[0])
        #print(num_t)
        #print(count_conv_fp.items)
        small_rank = np.argsort(value)[::-1]
        #print('rank value', value[small_rank])
        count_conv_fp[name] = small_rank[: num_t]
        #print('after count_conv_fp[name]', count_conv_fp[name])
        
    for (name, value) in count_fc_fp.items():
        #value = np.array(value)
        num_t = int(compress_rate * value.shape[0])
        #print(num_t)
        small_rank = np.argsort(value)[::-1]
        count_fc_fp[name] = small_rank[: num_t]
        
    Merge(count_conv_fp, count_fc_fp)
    return count_conv_fp
    


# In[14]:


def get_code(compress_rate, model_original, args, train_loader, test_loader, val_loader):
    model = copy.deepcopy(model_original)
    
    if args.cuda:
        model.cuda()
    #print(APoZ(conv_fp_dict, fc_fp_dict, count_conv_fp, count_fc_fp, num_t))
    return(APoZ(compress_rate, model, args, train_loader, test_loader, val_loader))
    #print(count_conv_fp, count_fc_fp)






