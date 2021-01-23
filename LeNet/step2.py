import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import copy
import pickle

from utils import *
from ec import *
from pruning import *
 

def recode2binary(integer_code, filter_nums, original_filter):
    res = []
    temp = []
    for i in range(len(integer_code)):
        #index表示取组合中的哪一部分，取值范围为0-49
        index = i
        #value表示采用的是（0-199）种组合中的哪一种
        value = integer_code[i][0]
        #print(i)
        #print(value)
        temp = original_filter[value][index]
        #希望得到的是filters_length * 1
        for j in (temp):
            res.append(j)
    
    res = np.array(res)
    res = res.reshape(sum(filter_nums), 1)
    #print(res)
    #print(res.shape)
    #print(res)
    return res

def evaIndividual(ind, model, args, train_loader, test_loader, val_loader, original_filter):
    gene_length = args.gene_length
    filter_nums = args.filter_nums 
    res_30 = 0
    
    solution = np.ones((sum(filter_nums), 1), dtype = np.int)
    integer_code = np.ones((gene_length, 1), dtype = np.int)
    #
    integer_code = ind.reshape(ind.shape[0], 1)
    #print(integer_code)
    #这行代码代表最后一个全连接层不进行剪枝，卷积核对应位始终是1，现在先不考虑
    #solution[-10:] = 1  # Last 10 output should not be changed
    res = recode2binary(integer_code, filter_nums, original_filter)
    solution = res.reshape(res.shape[0], 1)
    solution[-10:] = 1 
    # Prune model according to the solution
    model_new = prune_model(model, solution, args)
    
    #计算微调前的准确率
    #这一步待考量，因为计算微调前的准确率太耗时
    acc, loss = test_forward(val_loader, model_new, args)
    #打印微调前的准确率和损失
    
    pruning_rate = 1 - np.sum(solution) / (sum(filter_nums))
    
    print('step2微调前:  * accuracy {acc:.2f}, loss {loss:.2f}, pruning {pruning:.2f}'
         .format(acc = acc, loss = loss, pruning=pruning_rate))
    
    #计算微调后的准确率
    acc, loss, res_30 = train_forward(train_loader, test_loader, val_loader, model_new, args)  # test_forward(model_new)
    #acc, loss = test_forward(val_loader, model_new, criterion, solution)
    
    #print(acc, pruning_rate)
    #打印微调前的准确率和损失
    print('step2微调后:  * accuracy {acc:.2f}, pruning {pruning:.2f}'
         .format(acc = acc, pruning=pruning_rate))
    
    return 100-acc, np.sum(solution), res_30

class Individual():
    
    def __init__(self, gene_length, p_init, model, args, train_loader, test_loader, val_loader, original_filter):
        self.temp = 0
        self.dec = np.zeros(gene_length, dtype=np.uint8)  ## binary
        for i in range(gene_length):
            self.dec[i] = p_init[i]  # always begin with 1
        self.obj = [0, 0]  # initial obj value will be replaced by evaluate()
        self.evaluate(model, args, train_loader, test_loader, val_loader, original_filter) 

    def evaluate(self, model, args, train_loader, test_loader, val_loader, original_filter):
        self.obj[0], self.obj[1], self.temp = evaIndividual(self.dec, model, args, train_loader, test_loader, val_loader, original_filter)
        
def initialization(pop_size, gene_length, model, args, train_loader, test_loader, val_loader, original_filter, p0, target_dir):
    population = []
    acc_temp = []
    #count = 0
    for i in range(pop_size):
        ind = Individual(gene_length, p0, model, args, train_loader, test_loader, val_loader, original_filter)
        population.append(ind)
        acc_temp.append(ind.temp)
        path_save = './' + target_dir
        
        with open(path_save + "population_100_re.pkl", 'wb') as f:
            pickle.dump(population, f) 
            
        with open(path_save + "acc_temp_re.pkl", 'wb') as f:
            pickle.dump(acc_temp, f) 

        
    return population
           

def get_population(model, args, train_loader, test_loader, val_loader):
    
    original_filter = pickle.load(open('filter_dict/filter_dict.pkl', 'rb'))
    p0 = pickle.load(open('population_dict/population_dict.pkl', 'rb'))
    target_dir = 'Results2/'

    # configuration
    #种群数量先设为8
    pop_size = len(p0)  # Population size
    n_obj = 2  # Objective variable dimensionality
    filter_nums = args.filter_nums
    dec_dim = args.gene_length  # Decision variable dimensionality

    # Initialization
    population = initialization(pop_size, dec_dim, model, args, train_loader, test_loader, val_loader, original_filter, p0, target_dir)
    
                
    print('final population')
    for ind in population:
        
        pruning_rate = 1 - ind.obj[1] / (sum(filter_nums)) 
        
        print(' *accuracy {acc:.2f}, pruning {pruning:.2f}'
             .format(acc = 100 - ind.obj[0], pruning=pruning_rate))
     