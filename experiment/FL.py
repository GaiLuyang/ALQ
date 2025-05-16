import torch
import random
import bisect
import copy
from tqdm import tqdm
import QDP
import time
import math

def norm_l2(a_model_body, b_model_body): 
    if b_model_body == None:
        l2 = sum(p_client.norm(p=2) for p_client in a_model_body.parameters())
    else:
        l2 = sum((p_client - p_global).norm(p=2) for p_client, p_global in zip(a_model_body.parameters(), b_model_body.parameters()))
    return torch.sqrt(l2)


def clients_sample(users = None, 
                   weights = None, 
                   k = None):
    cumulative_weights = [0]
    for weight in weights:
        cumulative_weights.append(cumulative_weights[-1] + weight)

    result = []
    selected_indices = set()

    for _ in range(k):
        random_num = random.random()
        index = bisect.bisect(cumulative_weights, random_num) - 1

        while index in selected_indices:
            random_num = random.random()
            index = bisect.bisect(cumulative_weights, random_num) - 1

        selected_indices.add(index)
        result.append(users[index])
    return torch.tensor(result) 

def server_update_fedavg(clients_list = None,
                        train_clients_list = None,
                        server = None,
                        task = None):
    server.model.model_k = copy.deepcopy(server.model.model_body) 
    bit = 0 
    energy = 0 
    cal = 0 
    out = 0 
    for i in train_clients_list:
        bit += clients_list[i].model.param_num * task.b + 32
        clients_list[i].Ran = Q_Range(clients_list[i], clients_list[i].model.model_body, server.model.model_k, clients_list[i].model.param_num, task)
        energy += (task.tao ** 2) * (clients_list[i].loc[0]**2) * task.N0 * task.Bn * (2 ** ((task.b * clients_list[i].model.param_num + 32) / task.tao / task.Bn) -1)
        for param_server, param_server_model_k, param_client, param_client_server  in zip(server.model.model_body.parameters(), 
                                                                    server.model.model_k.parameters(),
                                                                   clients_list[i].model.model_body.parameters(), 
                                                                   clients_list[i].model.server_model.parameters(),):
            if torch.isnan(param_client).any() or torch.isinf(param_client).any():
                out += 1
                break
            cal3 = time.time()
            param = param_client.data - param_server_model_k.data
            param = QDP.Quantization(param, param_server_model_k.data * 0, clients_list[i].Ran, task)
            cal4 = time.time()
            cal += cal4 - cal3 
            param_server.data += 1 / (len(train_clients_list)-out) * param 
    return bit, energy, cal / task.client_need

def Fed_AVG(clients_list = None, 
            server = None, 
            task = None
                     ):
    LOSS_test = []
    acc = []
    model_bound = []
    bits = [] 
    up_bits = [] 
    cum_cost = [] 
    cal_cost = [] 
    total_cost = [] 
    energies = [] 
    loss_i, acc_i = server.model.model_pre(-1,task.server_batch_size)
    print('Loss:',loss_i)
    print('Acc:',acc_i)
    acc.append(acc_i)
    LOSS_test.append(loss_i)
    model_bound.append(0)
    bits.append(0)
    up_bits.append(0)
    cum_cost.append(0) 
    cal_cost.append(0) 
    total_cost.append(0) 
    energies.append(0)
    for i in tqdm(range(task.global_epoch), desc=task.fl+'_'+task.opt+'_'+task.exp_num, total=task.global_epoch):
        train_clients_list = clients_sample(users = list(range(len(clients_list))), 
                                             weights = task.client_probability, 
                                             k = task.client_need)
        down_bit = 0
        dowm_energy = 0
        for k in train_clients_list:   
            down_bit += clients_list[k].model.param_num * 32 
            dowm_energy += (task.tao ** 2) * (clients_list[k].loc[0]**2) * task.N0 * task.Bn *  (2 ** ((32 * clients_list[k].model.param_num ) / task.tao / task.Bn) -1)
            for param_server, param_client_server, param_client in zip(server.model.model_body.parameters(), 
                                                                       clients_list[k].model.server_model.parameters(), 
                                                                       clients_list[k].model.model_body.parameters()):
                param_client.data = copy.deepcopy(param_server.data)
                param_client_server.data = copy.deepcopy(param_server.data)
        cal1 = time.time()
        for j in train_clients_list:
            clients_list[j].model.model_train(train_epoch_num = i, way = 'fedavg', task = task)
        cal2 = time.time() 
        up_bit, up_energy, cal = server_update_fedavg(clients_list = clients_list,
                                    train_clients_list = train_clients_list,
                                    server = server,
                                    task = task
                                    )  
        loss_i, acc_i = server.model.model_pre(i, task.server_batch_size)
        print('Loss:',loss_i)
        print('Acc:',acc_i)
        if math.isnan(loss_i):
            print('last time')
            server.model.model_body = copy.deepcopy(server.model.model_k) 
        acc.append(acc_i)
        LOSS_test.append(loss_i)
        bits.append(bits[-1] + up_bit + down_bit)
        up_bits.append(up_bits[-1] + up_bit)
        cum = up_bit / task.Bn
        cum_cost.append(cum_cost[-1] + cum) 
        cal_cost.append(cal_cost[-1] + (cal2-cal1)/task.client_need + cal) 
        total_cost.append(total_cost[-1] + cum + (cal2-cal1)/task.client_need + cal) 
        energies.append(energies[-1] + up_energy + dowm_energy)

    return LOSS_test, acc, model_bound, bits, energies, up_bits, cum_cost, cal_cost, total_cost

def Fed_ADMM(clients_list = None, 
                     server = None, 
                     task = None, 
                     ):
    LOSS_test = []
    acc = []
    model_bound = []
    bits = [] 
    up_bits = [] 
    cum_cost = [] 
    cal_cost = [] 
    total_cost = [] 
    energies = [] 
    loss_i, acc_i = server.model.model_pre(-1,task.server_batch_size)
    print('Loss:',loss_i)
    print('Acc:',acc_i)
    acc.append(acc_i)
    LOSS_test.append(loss_i)
    model_bound.append(0)
    bits.append(0)
    up_bits.append(0)
    cum_cost.append(0) 
    cal_cost.append(0) 
    total_cost.append(0) 
    energies.append(0)
    for i in tqdm(range(task.global_epoch), desc=task.fl+'_'+task.opt, total=task.global_epoch):
        train_clients_list = clients_sample(users = list(range(len(clients_list))), 
                                             weights = task.client_probability, 
                                             k = task.client_need)
        down_bit = 0
        dowm_energy = 0
        for k in train_clients_list:  
            down_bit += clients_list[k].model.param_num * 32
            dowm_energy += (task.tao ** 2) * (clients_list[k].loc[0]**2) * task.N0 * task.Bn *  (2 ** ((32 * clients_list[k].model.param_num ) / task.tao / task.Bn) -1)
            for param_server, param_client in zip(server.model.model_body.parameters(), clients_list[k].model.server_model.parameters()):
                param_client.data = copy.deepcopy(param_server.data)
        cal1 = time.time()
        for j in train_clients_list:
            clients_list[j].model.model_train(train_epoch_num = i, way = 'admm', task = task)
        cal2 = time.time()
        up_bit, up_energy, cal = server_update_admm(clients_list = clients_list,
                                    train_clients_list = train_clients_list,
                                    server = server,
                                    task = task
                                    ) 

        loss_i, acc_i = server.model.model_pre(i, task.server_batch_size)
        print('Loss:',loss_i)
        print('Acc:',acc_i)
        if math.isnan(loss_i):
            print('last time')
            server.model.model_body = copy.deepcopy(server.model.model_k) 
        acc.append(acc_i)
        LOSS_test.append(loss_i)
        for k in train_clients_list: 
            clients_list[k].model.y_update()
        bits.append(bits[-1] + up_bit + down_bit)
        up_bits.append(up_bits[-1] + up_bit)
        cum = up_bit / task.Bn
        cum_cost.append(cum_cost[-1] + cum) 
        cal_cost.append(cal_cost[-1] + (cal2-cal1)/task.client_need + cal) 
        total_cost.append(total_cost[-1] + cum + (cal2-cal1)/task.client_need + cal) 
        energies.append(energies[-1] + up_energy + dowm_energy)
    return LOSS_test, acc, model_bound, bits, energies, up_bits, cum_cost, cal_cost, total_cost

def server_update_admm(clients_list = None, 
                       train_clients_list = None, 
                  server = None, 
                  task = None
                  ):
    server.model.model_k = copy.deepcopy(server.model.model_body) 
    bit = 0
    energy = 0 
    cal = 0
    out = 0 
    for i in train_clients_list:
        bit += clients_list[i].model.param_num * task.b + 32
        clients_list[i].Ran = Q_Range(clients_list[i], clients_list[i].model.model_body, server.model.model_k, clients_list[i].model.param_num, task)
        energy += (task.tao ** 2) * (clients_list[i].loc[0]**2) * task.N0 * task.Bn * (2 ** ((task.b * clients_list[i].model.param_num + 32) / task.tao / task.Bn) -1)
        for param_server, param_client, param_y, param_server_model_k in zip(server.model.model_body.parameters(), 
                                            clients_list[i].model.model_body.parameters(),
                                            clients_list[i].model.y.parameters(),
                                            server.model.model_k.parameters(),):
            if torch.isnan(param_client).any() or torch.isnan(param_y).any() or torch.isinf(param_client).any() or torch.isinf(param_y).any():
                out += 1
                break
            cal3 = time.time()
            param = param_client.data - param_server_model_k.data
            param = QDP.Quantization(param, param_server_model_k.data * 0, clients_list[i].Ran, task)
            param_client.data = param + param_server_model_k.data
            param = task.weight[i] * (param + param_y.data/task.rho[i])
            cal4 = time.time()
            cal += cal4 - cal3 
            param_server.data += 1 / (len(train_clients_list)-out) * param

    return bit, energy, cal / task.client_need

def Q_Range(client, client_model_body, server_model_k, param_num, task):
    if task.opt == 'ALQ' and task.fl == 'ADMM':
        m = 4 * (2**task.b-task.alpha) * (2**task.b-task.alpha-1) * param_num
        Ran =  task.ex * min(torch.sqrt((norm_l2(client_model_body, server_model_k)**2) * ((2**task.b - 1 - 2 * task.alpha)**2)/m).item(), client.Ran)
    elif task.opt == 'ALQ' and task.fl == 'AVG':
        Ma = (task.total_rounds-160.0)/(20.0 * torch.sqrt(task.total_rounds).item())-4.0 * (task.client_num-task.client_need)/(task.client_need*(task.client_num-1))
        if Ma <= 0:
            raise ValueError("Range error, change T, n and r.")
        Mb = task.client_num * task.client_need * (task.client_num-1) * norm_l2(client_model_body, server_model_k)**2 / (4*task.client_num**2-3*task.client_num*task.client_need-task.client_need)
        Mc = ((2**task.b - 1 - 2 * task.alpha)**2)/(4 * (2**task.b-task.alpha) * (2**task.b-task.alpha-1) * param_num)
        Ran =  task.ex * min(torch.sqrt(Ma * Mb * Mc).item(), client.Ran)
    return Ran

def Fed_fl(clients_list, server, task):
    if task.fl == 'ADMM':
        LOSS_test, Acc, model_bound, Bits, energies, up_bits, cum_cost, cal_cost, total_cost\
                                    = Fed_ADMM(
                                    clients_list = clients_list, 
                                    server = server, 
                                    task = task
                                    )
    if task.fl == 'AVG':
        LOSS_test, Acc, model_bound, Bits, energies, up_bits, cum_cost, cal_cost, total_cost\
                                    = Fed_AVG(
                                    clients_list = clients_list, 
                                    server = server, 
                                    task = task
                                    )
    return LOSS_test, Acc, model_bound, Bits, energies, up_bits, cum_cost, cal_cost, total_cost