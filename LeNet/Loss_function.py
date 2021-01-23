import torch
from torch.autograd import Variable
from torchvision import models
#import cv2
#import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from operator import itemgetter
from heapq import nsmallest
import time

from torchvision import datasets, transforms
import copy
import random
    
class FilterPrunner:
    def __init__(self, model, args):
        self.model = model
        self.reset()

    def reset(self):
        #大字典，存储每一个卷积层的卷积核排名
        self.filter_ranks = {}
    
    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        
        activation_index = 0
        
        #这里可能有问题
        for layer, (name, module) in enumerate(self.model._modules.items()):
            x = module(x)
            x = F.relu(x)   
            #找到卷积层
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
     
                self.activations.append(x)#.data.cpu().numpy())
                
                self.activation_to_layer[activation_index] = name
                activation_index += 1
    
            if layer < 2:
                x = F.max_pool2d(x, 2, 2)
                
        x = x.view(-1, 10)
        return x 
    
    #计算每一层的卷积核的Taylor值存进对应的字典filter_ranks
    def compute_rank(self, grad):
 
        activation_index = len(self.activations) - self.grad_index - 1
        
        activation = self.activations[activation_index]
        taylor = activation * grad
        
        #只有Torch张量才能这样取mean，第一维是batch，第二维是卷积核个数，第三维，第四维是内核
        taylor = taylor.mean(dim = (0, 2, 3)).data
        
        #假如这一层的卷积核的Taylor还没有装进filters_ranks，则存储
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()
            
            if torch.cuda.is_available():
                #print('!!')
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
                
            
        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1
        
        
    
    def lowest_ranking_filters(self, compress_rate):
        res = []
        #取出每一层对应的排序后的feature map
        for i in sorted(self.filter_ranks.keys()):
            
            count = 0
            data = []
            num = int(compress_rate * count)
            #计算每一层feature map的个数
            for j in range(self.filter_ranks[i].size(0)):
                count += 1
                #第一维是对应的层数，第二维是对应的卷积核，第三维是卷积核对应的Taylor数
                a = self.activation_to_layer[i]
                data.append((a, j, self.filter_ranks[i][j]))
                
            num = int(compress_rate * count)
            pruning_idx = nsmallest(num, data, itemgetter(2))
            
            res.append(pruning_idx)
            
        #print(num)
        #返回的是data数组中最小的num个数，排序标准是根据第三维的大小，即Taylor值大小
        return res
        
    def normalize_ranks_per_layer(self):
        #self.filter_ranks.cpu()
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v
            
    
#返回一个字典，key值是对应的卷积层名，value是对应的该层会被剪枝的filters下标
def get_code(compress_rate, model_original, args, train_loader, test_loader, val_loader):
    
    model = copy.deepcopy(model_original)
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    eval_acc = 0
    eval_loss = 0
    res_lf = {}
    
    #把module设成training模式
    prunner = FilterPrunner(model, args)
    model.train()
    
    for batch_idx, (data, target) in enumerate(val_loader):
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #Variable类对Tensor对象进行封装，会保存该张量对应的梯度，以及对生成该张量的函数grad_fn的一个引用。
        #如果该张量是用户创建的，grad_fn是None，称这样的Variable为叶子Variable。
        data, target = Variable(data), Variable(target)
        
    
        optimizer.zero_grad()
        
        prunner.reset()
        
        output = prunner.forward(data)
        #output = model(data)
        
        loss = F.cross_entropy(output, target)
        #loss = F.CrossEntropyLoss(output, target)

        eval_loss += loss.item() * target.size(0)
        #print('eval_loss', eval_loss)
        _, pred = torch.max(output, 1)
        num_correct = (pred == target).sum()
        eval_acc += num_correct.item()
        
        loss.backward()
        optimizer.step()
        
    
    res = {}
    #prunner.normalize_ranks_per_layer()
    result = prunner.lowest_ranking_filters(compress_rate)
    
    name = args.name
    
    for i in name:
        res[i] = []

   
    for i in range(len(result)):
        temp = result[i]
        
        for j in temp:
            
            res[j[0]].append(j[1])
    
    prunner.filter_ranks = {}
    prunner.activations = []
    prunner.gradients = []
    #self.grad_index = 0
    prunner.activation_to_layer = {}
    return res
    


        