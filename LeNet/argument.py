class Args:
    def __init__(self):
    
        self.lr = 0.1
        self.ft_lr = 0.0001
        self.retrain_lr = 0.1
        self.epochs = 160
        self.ft_epochs = 100
        self.temp_epoch = 30
        self.batch_size = 64
        self.test_batch_size = 500
        self.val_batch_size = 500
        self.momentum = 0.9
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.weight_decay = 1e-4
        self.filter_nums = [20, 50, 500, 10] 
        self.conv_layer = ['conv1', 'conv2', 'conv3', 'conv4']
        self.name = ['conv1', 'conv2', 'conv3', 'conv4']
        self.gene_length = 30
        self.filter_length = 580
        self.accuracy = 93.7
        self.train = False


        