import torch
import numpy as np
import participant
import dataload 
import divide
import FL
import torch.nn as nn
import TASK
import copy
import time
import savefile
from config import *
torch.manual_seed(1)
################################ Initialization parameter setting menu ##################################
task = TASK.Task(
############ Basic setting parameters ############
                device = "cuda:0", 
                fl = 'ADMM', 
                opt = 'ALQ', 
                machine = 'notebook', 
                exper = EXPER,
                exp_num = TIMES,# number of experiments 
                choose = 1,# data set selection 
                train_rate = 0.7, 
############ algorithm-specific parameter ############
                b = 3, # bit quantity of a single parameter (quantization bits) 
                alpha = torch.tensor(2), # quantization range multiple hyperparameter  
                eps_step = 1.8, # Privacy Budget Per Step 
                hyper = -5, # Hyperparameter for adjusting the ratio of left and right parameters 
                ex = 1, 
############ algorithm shared parameters ############    
                client_num = 100, # Initialize client count 
                client_need = 100 ,  # Number of training clients required per round 
                global_epoch = 1400,  # Number of global iterations  
                local_epoch = 30, # number of local iterations 
                lr = torch.tensor(0.09), 
                momentum = torch.tensor(0.9), 
                weight_decay = torch.tensor(0.0001),  
                y_lr = torch.tensor(0.5), # Dual variable update learning rate of admm 
                noniid_level = NONIID, # degree of heterogeneity 
                batch_size = 16, # Local batch training size 
                server_batch_size = 'all', # Server Predicted Batch Size 
                lambda_l2 = torch.tensor(0.0000), 
                rho = torch.tensor(0.1), # penalty parameter 
                model_name = 'Cifar10_CNN_Net',            
                criterion = nn.CrossEntropyLoss(), 
############ federal simulation parameter ############
                cum_cal_rate = 0.01, 
                radius = 10000, 
                N0 = 0.000001, 
                tao = 200, 
                band_sum = 5000000 
                )
probability_end = 2.5
probability = np.linspace(1,probability_end,task.client_num)
task.client_probability = probability/sum(probability)

task.weight=torch.ones(task.client_num, dtype=torch.float32)

for i,prob in enumerate(task.client_probability):
    if prob>=1/task.client_num:
        task.weight[i] = torch.tensor(1, dtype=torch.float32).to(task.device)
    else:
        task.weight[i] = torch.tensor(1, dtype=torch.float32).to(task.device)
task.file_name = savefile.create_file(task)
################################### data loading ####################################
data_file_name=['mnist', 'cifar10']
start = []
end = []

start.append(time.time())  
datasets_train, datasets_test = dataload.dataloading(data_file_name[task.choose]) 
task.datasets_train = datasets_train
end.append(time.time()) 
print('1.Data loaded done. Run: {:.4}s'.format(end[0] - start[0]))
# ################################ data pre-processing #################################### 
start.append(time.time()) 
clients_data_index, datasets_train, datasets_test =divide.dirichlet_split_noniid(datasets_train, 
                                                datasets_test,
                                                task
                                                )
end.append(time.time()) 
print('2.Dirichlet split done. Run: {:.4}s'.format(end[1] - start[1]))
################################ initialization tasks ##################################
start.append(time.time()) 
server = participant.Server(
                            datasets_test = datasets_test,
                            task = task
                            )
server.model.model_body.to(task.device)
clients_list = participant.create_clients(
                                        clients_data_index = clients_data_index,
                                        datasets_train = datasets_train,
                                        server_model = copy.deepcopy(server.model.model_body), 
                                        y = copy.deepcopy(server.y),
                                        task = task
                                        )
for i in range(len(clients_list)):
    clients_list[i].model.model_body.to(task.device)
    clients_list[i].model.y.to(task.device)
end.append(time.time()) 
print('3.Set task done. Run: {:.4}s'.format(end[2] - start[2]))
###############################FL training#######################################
start.append(time.time())  
LOSS_test, Acc, model_bound, Bits, energies, up_bits, cum_cost, cal_cost, total_cost\
                            = FL.Fed_fl(clients_list = clients_list, 
                            server = server, 
                            task = task)
end.append(time.time())  
print('4.FL train done. Run: {:.4}s'.format(end[3] - start[3]))
############################### file storage ##############################
savefile.save_result(LOSS_test, Acc, model_bound, Bits, energies, up_bits, cum_cost, cal_cost, total_cost, task)


