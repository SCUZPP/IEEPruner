class Args:
    def __init__(self):
    
        self.depth = 19
        #kgea = False
        self.constraint = False
        
        #此rand是指对两种不对全连接层剪枝的算法随机产生剪枝下标
        self.rand = True
        
        #指重新训练
        self.retrain = False
        
        self.lr = 0.1
        self.ft_lr = 0.0001
        self.retrain_lr = 0.1
        self.epochs = 160
        self.batch_size = 64
        self.test_batch_size = 500
        self.val_batch_size = 500
        self.momentum = 0.9
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.weight_decay = 1e-4
        #最后三层是全连接层
        if self.depth == 16:
        
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 40
            self.filter_nums = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
            self.fc_nums_dict = { '1': 4096, '4': 4096, '6': 10}
            self.conv_layer = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 1, 4, 6]
            self.fc_layer = [1, 4, 6]
            self.conv_name = ['0', '3', '7', '10', '14', '17', '20', '24', '27', '30', '34', '37', '40']
            self.name = ['0', '3', '7', '10', '14', '17', '20', '24', '27', '30', '34', '37', '40', '1', '4', '6']
            self.fc_name = ['1', '4', '6']
            self.conv_name_dict = {'0': [], '3': [], '7': [], '10': [], '14': [], '17': [], '20': [], '24': [], '27': [], '30': [], '34': [], '37': [], '40': []}
            self.fc_name_dict = { '1': [], '4': [], '6': []}
            self.gene_length = 50
            self.filter_length = 12426
            self.accuracy = 93.7
        
        else:
            self.ft_epochs = 300
            self.re_epochs = 300
            self.temp_epoch = 60
            self.filter_nums = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 4096, 4096, 10]
            self.conv_layer = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49, 1, 4, 6]
            self.fc_nums_dict = { '1': 4096, '4': 4096, '6': 10}
            self.fc_layer = [1, 4, 6]
            self.name = ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49', '1', '4', '6']
            self.fc_name = ['1', '4', '6']
            self.gene_length = 50
            self.filter_length = 13706
            self.accuracy = 93.84

        