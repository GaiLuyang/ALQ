import torch
import torch
import torch.optim as optim
import copy
import time
import gc

class Models_load():
    def __init__(self, 
                 model_name = None, 
                 lr = None, 
                 client_dataset = None, 
                 client_data_index = None,
                 server_dataset = None, 
                 local_epoch = None, 
                 model_body = None, 
                 criterion = None, 
                 rho_k = None, 
                 eta_k = None,
                 batch_size = None, 
                 lambda_l2 = None,
                 y = None,
                 y_lr = None,
                 task = None):
        self.task = task
        self.model_name = model_name
        self.client_dataset = client_dataset
        self.client_data_index = client_data_index
        self.server_dataset = server_dataset
        if server_dataset != None:
            self.data_num = len(server_dataset)
        if client_dataset != None:
            self.data_num = len(client_dataset)
        self.lr = lr
        self.batch_size = batch_size 
        self.local_epoch = local_epoch
        self.rho_k = rho_k
        self.eta_k = eta_k
        self.lambda_l2  = lambda_l2 
        self.model_body = copy.deepcopy(model_body) 
        self.param_num = sum(p.numel() for p in self.model_body.parameters()) 
        self.criterion = criterion
        self.server_model = copy.deepcopy(model_body)
        self.y = y 
        self.y_lr = y_lr
        self.train_epoch_num = [-1]
        self.delay_num = 0
        self.model_app = copy.deepcopy(model_body) 
        self.train_num = 0 
        self.optimizer = optim.SGD(self.model_body.parameters(), lr=self.lr, momentum = task.momentum)



    def model_train(self, train_epoch_num = None, way = None, task = None):
        self.train_num += 1 
        self.model_k = copy.deepcopy(self.model_body)
        self.train_epoch_num.append(train_epoch_num) 
        self.delay_num = self.train_epoch_num[-1]-self.train_epoch_num[-2]-1
        k=0 
        for epoch in range(1000): 
            torch.manual_seed(time.time()*1000)
            if self.batch_size == 'all':
                self.batch_size = self.data_num
            train_loader = torch.utils.data.DataLoader(self.client_dataset, batch_size=self.batch_size, shuffle=True)
            for batch_idx,(train_x, train_label) in enumerate(train_loader):
                train_x = train_x.to(self.task.device)
                self.model_body.train()
                self.optimizer.zero_grad()
                train_pre=self.model_body(train_x)
                train_label_onehot = onehot(train_label)
                train_label_onehot = train_label_onehot.to(self.task.device)
                if way == 'admm':
                    if task.opt == 'DPadmm':
                        loss = self.criterion(train_pre, train_label_onehot)
                        self.eta_k = 1/(task.c3 + self.lambda_l2 * task.c4 / task.client_num + 4 * task.c1 *torch.sqrt(self.param_num * train_epoch_num *torch.log(1.25 / task.delta))/(self.data_num * task.eps_step * task.cw))
                        loss.backward()
                        self.model_update_dpadmm()
                    else:
                        loss = self.criterion(train_pre, train_label_onehot) 
                        loss.backward()
                        self.model_update_admm()
                elif way == 'fedavg':
                    loss = self.criterion(train_pre, train_label_onehot) 
                    loss.backward()
                    self.model_update_fedavg()
                if k == self.local_epoch-1:
                    break 
                else:
                    k += 1
            if k == self.local_epoch-1:
                break 
            torch.cuda.empty_cache()
        del self.model_k
        gc.collect()
            
    def y_update(self):
        for client_param, server_param, y_param in zip(self.model_body.parameters(), self.server_model.parameters(), self.y.parameters()):
            y_param.data = y_param.data + self.y_lr * self.rho_k * (client_param.data - server_param.data)
        
    def model_pre(self, i = None, server_batch_size = None):
        torch.manual_seed(time.time()*1000) 
        if server_batch_size == 'all':
            server_batch_size = self.data_num
        test_loader = torch.utils.data.DataLoader(self.server_dataset, batch_size = server_batch_size, shuffle = True)
        self.model_body.eval() 
        total_correct=0 
        total_loss=0 
        for batch_idx,(test_x, test_label) in enumerate(test_loader): 
            test_x = test_x.to(self.task.device)
            test_outputs = self.model_body(test_x) 
            test_label_onehot = onehot(test_label)
            test_label_onehot = test_label_onehot.to(self.task.device)
            l2_reg = self.lambda_l2 * sum(p.pow(2.0).sum() for p in self.model_body.parameters())
            test_loss = self.criterion(test_outputs, test_label_onehot) + l2_reg
            total_loss += test_loss

            pred = test_outputs.argmax(dim=1) 
            test_label = test_label.to(self.task.device)
            correct = pred.eq(test_label).sum().float().item() 
            total_correct+=correct
            break
        total_num = len(test_loader.dataset) 
        test_acc = total_correct / server_batch_size
        test_loss = total_loss 
        return test_loss.item(), test_acc

    def model_to_zero(self):
        for name, param in self.model_body.named_parameters():
            param.data = param.data * 0
    
    def model_update_admm(self):
        for client_param, y_param, server_param in zip(self.model_body.parameters(), self.y.parameters(), self.server_model.parameters()):
            client_param.data = client_param.data - self.lr * (client_param.grad.data + y_param.data + self.rho_k * (client_param.data - server_param.data))

    def model_update_dpadmm(self): 
        for client_param, client_last_param, y_param, server_param in zip(self.model_body.parameters(), self.model_k.parameters(), self.y.parameters(), self.server_model.parameters()):
            client_param.data = client_param.data - self.lr * (client_param.grad.data + y_param.data + self.rho_k * (client_param.data - server_param.data) + (client_param.data - client_last_param.data)/self.eta_k)

    def model_update_fedavg(self):
        for client_param in self.model_body.parameters():
            client_param.data = client_param.data - self.lr * client_param.grad.data

def onehot(label,depth=10):
    out=torch.zeros(label.size(0),depth)
    idx=torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out