import ABS
import APoZ
import Geometric_median
import Loss_function
import numpy as np
import pickle
import torch
import logging

import torch.optim as optim
logger = logging.getLogger("__main__")
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#filter_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
#染色体长度是30，4224/50 = 84，则基因的每一位代表84个卷积核, 最后一条染色体代表84 + 24个filters
#所以通过某种方法得到的filters下标需要处理成50段

#处理filters组合，将其分为50段

def handle_code(pruning_filter, args):

    #print(pruning_filter)
    name = args.name
    gene_length = args.gene_length
    filter_num = args.filter_nums
    filter_length =sum(filter_num)
    pruning_filter_index = []
    
    index = 0
    count = 0
    #四种算法都是不是对每一层剪枝相同的数量的filters，所以存在第一层过了是第三层这种错误
    #把loss function算法改写，对应没有filters被剪枝的层，value被设为空，这样就不会存在下标错误
    '''for (layer, value) in (pruning_filter.items()):
        
        if index == 0:
            for i in value:
                pruning_filter_index.append(i)
                
        else:
            count += filter_num[index - 1]
            for i in value:
                temp = count + i
                pruning_filter_index.append(temp)
                
            
        index = index + 1
 '''
   # name 按顺序存储卷积层和全连接层的下标，按照name中的顺序取出层对应的filters
        
    for name_layer in name:
        temp_filters = pruning_filter[name_layer]
        if index == 0:
            for filters in temp_filters:
                pruning_filter_index.append(filters)
                
        else:
            count += filter_num[index - 1]
            for filters in temp_filters:
                temp = count + filters
                pruning_filter_index.append(temp)   
                
        index = index + 1
            
    pruning_filter_index = sorted(pruning_filter_index)
    #print(pruning_filter_index)
    
    '''for (layer, value) in pruning_filter.items():
        for i in value:
            pruning_filter_index.append(i)
    pruning_filter_index = np.array(pruning_filter_index)
    print(pruning_filter_index)
    #把filters的下标按大小顺序排列
    #这个地方不对，因为原始是按每一层进行选择下标，所以有重复的地方会被排掉
    pruning_filter_index = sorted(pruning_filter_index)'''
    
    #int是往下取整数，所以最后一个基因要存储多余出来的filters
    #或者往上取整可以尝试，向上取整基因长度可能小于30不能向上
    pivot = int(filter_length / gene_length)
    #把基因的每一位初试化为1，1代表保留，0代表剪枝
    #这里表示每一列存储一个filters，为什么不是1行来存储？
    filter_code = np.ones((filter_length, 1), dtype=np.int)
    
    #出现在Pruning_filter_index中的filters的值被设为0
    for i in pruning_filter_index:
        filter_code[i] = 0
        
    #把filters化分成gene_length份，最后一位基因要存储所有剩下的filters
    #格式也是filters_nums * 1
    filter_divide = []
    j = 0
    for i in range(0, filter_length, pivot):
        if j == (gene_length - 1):
            low = j * pivot
            high = filter_length
            filter_divide.append(filter_code[low : high])
            break

        else:
                low = j * pivot
                high = (j + 1) * pivot
                filter_divide.append(filter_code[low : high])

        j = j + 1
        
    return filter_divide


#得到100种filters组合，这里应该是初始化种群时用，对照时也需要用
def original_code(model, args, train_loader, test_loader, val_loader):
    #剪枝率范围：20% - 70%
    #剪枝方法：四种
    #选50个剪枝率，分别用四种方法剪枝，得到200个filters组合

    #表示组合的下标
    index = 1

    #用列表来存储200个filters组合
    
    #种群数量初始为4，方便计算
    '''a = {'conv1': [9, 3], 'conv2': [18, 13, 11,  5], 'fc1': [27,  8, 38, 20, 11, 22, 15, 39, 25, 43], 'fc2': [6, 1]}
    b = {'conv1': [5, 3, 6], 'conv2': [18, 13, 11,  5, 4, 7], 'fc1': [23,  9, 38, 20, 14, 21, 15, 39, 35, 43, 40, 41], 'fc2': [4, 5, 6, 7]}
    c = {'conv1': [9, 1, 2, 3], 'conv2': [18, 15, 11,  4, 5, 6, 7, 8], 'fc1': [27,  9, 38, 22, 10, 23, 15, 39, 29, 43, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12], 'fc2': [4, 5, 6, 7]}
    d = {'conv1': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'conv2': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], 'fc1': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44], 'fc2': [0, 1, 2, 3, 4, 5, 6, 7, 8]}

    
    d = {'0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 
         '5': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
         '2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 
         '7': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 
         '10': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
         '12': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 
         '14': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '17': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  '19': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
         '21': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '24': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '26': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
         '28': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '4': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
         '6': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    
    b = {'0': [], '2': [], '5': [], '7': [], '10': [], '12': [], '14': [], '17': [],  '19': [], '21': [], '24': [], '26': [], '28': [], '1': [], '4': [], '6': []}
    
    c = {'0': [], '2': [], '5': [], '7': [], '10': [], '12': [], '14': [], '17': [],  '19': [], '21': [], '24': [], '26': [], '28': [], '1': [], '4': [], '6': []}
    
    a = {'0': [], '2': [], '5': [], '7': [], '10': [], '12': [], '14': [], '17': [],  '19': [], '21': [], '24': [], '26': [], '28': [], '1': [], '4': [], '6': []}
    
    pruning_filters = []
    temp = np.array(handle_code(d))
    #print(np.sum(temp[0]))
    pruning_filters.append(handle_code(d))
    pruning_filters.append(handle_code(b))
    pruning_filters.append(handle_code(c))
    pruning_filters.append(handle_code(a))
    
    #pickle.dump(pruning_filters, open('filter_dict.pkl', 'wb'))
    
    return pruning_filters
    
    
    '''
    
    #存储函数接口
    #func_list = [APoZ.get_code, ABS.get_code, Geometric_median.get_code, Loss_function.get_code]
    
    pruning_filters = []
    #种群数量初始为200
    if args.cuda: 
        low = 30
        high = 80
        
    else:
        low = 30
        high = 31
        
    #for i in range(30, 31):
    for i in range(low, high):
        print(i)
        logger.info('low_high_{}'.format(i))
        #i = np.random.randint(20, 71)
        i = i / 100
        
        #print('APoZ')
        pruning_filter = APoZ.get_code(i, model, args, train_loader, test_loader, val_loader)
        temp = handle_code(pruning_filter, args)
        #print(temp)
        pruning_filters.append(temp)
        #print('ABS')
        pruning_filter = ABS.get_code(i, model, args, train_loader, test_loader, val_loader)
        temp = handle_code(pruning_filter, args)
        #print(temp)
        pruning_filters.append(temp)
        
        #print('Geometric_median')
        pruning_filter = Geometric_median.get_code(i, model, args, train_loader, test_loader, val_loader)
        temp = handle_code(pruning_filter, args)
        #print(temp)
        pruning_filters.append(temp)

        pruning_filter = Loss_function.get_code(i, model, args, train_loader, test_loader, val_loader)
        temp = handle_code(pruning_filter, args)
        #print(temp)
        pruning_filters.append(temp)
        #"""
           
    if args.depth == 16:
        if args.rand:
            pickle.dump(pruning_filters, open('filter_dict/filter_dict_vgg16_rand_fc.pkl', 'wb')) 
        else:
            pickle.dump(pruning_filters, open('filter_dict/filter_dict_vgg16.pkl', 'wb')) 
            
    if args.depth == 19:
        if args.rand:
            pickle.dump(pruning_filters, open('filter_dict/filter_dict_vgg19_rand_fc.pkl', 'wb')) 
        else:
            pickle.dump(pruning_filters, open('filter_dict/filter_dict_vgg19.pkl', 'wb')) 
               






