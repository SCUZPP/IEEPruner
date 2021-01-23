import numpy as np
import copy
import torch
import torch.nn as nn

#self.filter_nums = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
def masking(solution, args):
    
    filter_nums = args.filter_nums
    solution = np.squeeze(solution)
    res = []
    low = 0
    high = filter_nums[0]
    #print('solution', solution.shape)
    for index in range(len(filter_nums)):
        
        if index == 0:
            filters = solution[low : high]
        else:
            low = high
            high = filter_nums[index] + low
            filters = solution[low : high]#[0]
            
        res.append(filters)
    #print('res', res)   
    return res
    
def prune_model(model_original, solution, args):
    model = copy.deepcopy(model_original)
    newmodel = copy.deepcopy(model_original)
    
    if args.cuda:
        model.cpu()
        newmodel.cpu()
    
    #把一维矩阵根据卷积层改为二维矩阵
    cfg_mask = masking(solution, args)
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1, ))
            #m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            #m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            #m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            #m1.running_var = m0.running_var[idx1.tolist()].clone()  
            
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            #print('m1.runing_var', m1.running_var.shape)
            #print('m0.running_var', m0.running_var.shape)
            #print('m1.running_mean', m1.running_mean.shape)
            #for i, (name, para) in enumerate(newmodel.state_dict().items()):
            #   if 'running_var' in name:
            #        print(name, para.shape)
            layer_id_in_cfg += 1
            start_mask = end_mask
            
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
            
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            
            #print('idx1_endmask', end_mask.shape, np.argwhere(np.asarray(end_mask)).shape, idx1.shape)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1, ))
                
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1, ))
            
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            
        elif isinstance(m0, nn.BatchNorm1d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        
        elif isinstance(m0, nn.Linear):
            #The first layer of the fully connected layer
            if layer_id_in_cfg == args.depth - 4:
                idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[layer_id_in_cfg])))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                    
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1, ))
            
                
                #全连接层输入，输出都改变
                w1 = m0.weight.data[:, idx0].clone()
                w1 = w1[idx1, :]
                m1.weight.data = w1.clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask
            
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            
            #默认不对全连接层后面的层剪枝，这里要重写
            else:
                
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1, ))

                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1, ))


                w1 = m0.weight.data[:, idx0].clone()
                w1 = w1[idx1, :]
                m1.weight.data = w1.clone()
                #print('m0.bias.data', m0.bias.shape)
                #print('m0.bias.data', m0.bias.data[idx1.tolist()].shape)
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask

                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg] 
                    
    #for i, (name, para) in enumerate(newmodel.named_parameters()):
    #    print(name, para.shape)
        
    return newmodel