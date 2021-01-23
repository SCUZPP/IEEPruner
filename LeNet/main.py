import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
import argument
from net import *
import original_coding 
import step1 
import step2 
import logging




#load data
args = argument.Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
logger = logging.getLogger(__name__)

basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(basic_format)    
log_path = 'log.txt' 
print('log_path {}'.format(log_path))
handler = logging.FileHandler(log_path, 'a', 'utf-8')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info('log_path {}'.format(log_path))
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_set = datasets.MNIST('../data', train = True, download = True,
                          transform = transforms.Compose([transforms.ToTensor(), 
                                                          transforms.Normalize((0.1307,), (0.3081,))]))


train_set, valset = torch.utils.data.random_split(train_set, [54000, 6000])
    

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.val_batch_size, shuffle=True, **kwargs)

testset = datasets.MNIST('../data', train = False, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)


#加载模型和数据
def construct_model_load():
    model = Net()
    '''
    if args.cuda:
        checkpoint = torch.load('../logs/conv2_cnn.pkl')

    else:
        checkpoint = torch.load('../logs/conv2_cnn.pkl', map_location=torch.device('cpu'))  

    model.load_state_dict(checkpoint)
    
    return model
    '''


model = construct_model_load()



#original_coding.original_code(model, args, train_loader, test_loader, val_loader)

#step1.get_population(model, args, train_loader, test_loader, val_loader)

step2.get_population(model, args, train_loader, test_loader, val_loader)
    

