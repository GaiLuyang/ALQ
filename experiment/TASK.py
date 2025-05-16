import torch
import numpy as np
import time
class Task():
    def __init__(self, 
                 exper,
                 exp_num,
                 choose, 
                 client_num, 
                 model_name, 
                 client_need, 
                 global_epoch, 
                 local_epoch, 
                 criterion,
                 lr, 
                 momentum, 
                 weight_decay,
                 rho, 
                 noniid_level,
                 batch_size, 
                 lambda_l2,
                 train_rate, 
                 server_batch_size, 
                 cum_cal_rate, 
                 y_lr = None,
                 fl = None,
                 b = None,
                 b1 = None,
                 b2 = None,
                 eta = torch.tensor(0), 
                 c1 = None,
                 alpha = None, 
                 global_lr = None,
                 device = None,
                 opt = None, 
                 machine = None,
                 eps_step = None,
                 APVP_eps = None,
                 Lap_param_eps = None,
                 delta = None,
                 hyper = None,
                 radius = None, 
                 N0 = None, 
                 tao = None, 
                 band_sum = None,
                 norm_bound = None, 
                 proportion = None, 
                 precision = None, 
                 ex = None,
                 SQ_Ran = None,
                 GSQ_Ran = None,
                 RQM_Ran = None,
                 p = None,
                 RQM_ex = None,
                 cw = None,
                 c3 = None,
                 c4 = None,
                 ):
        print('process on ' + device)
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.str_a = '############ basic setting parameters ############'
        self.device = device
        self.fl = fl
        self.opt = opt
        self.machine = machine
        self.exper = exper
        self.exp_num = exp_num
        self.choose = choose
        self.train_rate = train_rate
        self.str_b = '############ algorithm-specific parameters ############'
        self.p = p
        self.GSQ_eps = 0
        self.GSQ_Ran = GSQ_Ran 
        self.b = b
        self.alpha = alpha 
        self.hyper = hyper 
        self.ex = ex 
        self.eps_sum = 0 
        self.eps_step = eps_step 
        self.eps_record = 0 
        self.delta = delta 
        self.dp_times = 0 

        self.proportion = proportion 
        self.precision = precision 
        self.b1 = b1 
        self.b2 = b2 
        self.APVP_eps = APVP_eps 
        self.Lap_param_eps = Lap_param_eps 
        self.norm_bound = norm_bound 
        self.Lap_num = 0 
        self.APVP_eps = 0 
        self.Lap_eps = 0 
        self.min_loss = 100 
        self.c1 = c1 
        self.c3 = c3
        self.c4 = c4
        self.cw = cw
        self.eps_avg = 0 
        self.str_c = '############ algorithm shared parameters ############'
        self.client_num = client_num   
        self.client_need= client_need 
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch   
        self.total_rounds = torch.tensor(self.global_epoch * self.local_epoch)
        self.lr = lr.to(device)
        self.momentum = momentum.to(device)
        self.weight_decay = weight_decay.to(device)
        self.y_lr = y_lr 
        self.noniid_level = noniid_level
        self.batch_size = batch_size
        self.server_batch_size = server_batch_size
        self.lambda_l2 = lambda_l2.to(device)
        self.rho = [rho.to(self.device)]*self.client_num
        self.model_name = model_name
        self.criterion = criterion
        self.str_d = '############ federal simulation parameters ############'
        self.radius = radius 
        self.N0 = N0 
        self.tao = tao 
        self.band_sum = band_sum 
        self.Bn = self.band_sum / self.client_need
        torch.manual_seed(42)
        self.loc = torch.cat((torch.rand(self.client_num, 1) * self.radius, torch.rand(self.client_num, 1) * 2 * torch.pi), dim=1)
        self.cum_cal_rate = cum_cal_rate
        torch.manual_seed(time.time()*1000) 
        self.client_probability = None
        self.weight = None
        self.global_lr = global_lr
        self.eta = [eta.to(self.device)]*self.client_num
        self.file_name = None
        self.sigma_L = None 
        self.sigma_R = None 
        self.str_e = '############ other parameters ############'
        self.input_size = None





        
        
        
