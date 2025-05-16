import Model
import Net
import torch
import copy
import time
from tqdm import tqdm
class client():
    def __init__(self, 
                 client_data_index = None, 
                 client_dataset = None, 
                task = None,
                rho_k = None, 
                eta_k= None,
                loc = None,
                server_model = None,
                y = None):
        self.loc = loc 
        self.client_data_index = client_data_index
        self.client_dataset = client_dataset
        self.criterion = task.criterion 
        self.model_name = task.model_name
        self.local_epoch = task.local_epoch
        self.lr = task.lr 
        self.eta_k = eta_k 
        self.Ran = 1000 
        self.model = Model.Models_load(model_name = self.model_name, 
                              lr = self.lr,
                              client_dataset = self.client_dataset, 
                              client_data_index = self.client_data_index,
                              local_epoch = self.local_epoch, 
                              model_body = server_model, 
                              criterion = task.criterion,
                              rho_k = rho_k, 
                              eta_k = eta_k, 
                              batch_size = task.batch_size,
                              lambda_l2 = task.lambda_l2,
                              y = y,
                              y_lr = task.y_lr,
                              task = task
                              )

class Server():
    def __init__(self, 
                 datasets_test = None, 
                 task = None, 
                 ):
        self.loc = torch.tensor([0,0]),
        self.datasets_test = datasets_test
        self.label_num = len(torch.unique(datasets_test.targets))  
        if task.choose == 0:
            self.input_size = datasets_test.data.size()[1]*datasets_test.data.size()[2]
        self.model_name = task.model_name
        self.server_batch_size = task.server_batch_size
        self.criterion = task.criterion
        self.model_k = None 
        if self.model_name == 'Mnist_CNN_Net':
            torch.manual_seed(42) 
            self.model=Model.Models_load(model_name = self.model_name, 
                                server_dataset = self.datasets_test, 
                                model_body = Net.Mnist_CNN_Net(), 
                                criterion = self.criterion,
                                lambda_l2 = task.lambda_l2,
                                lr = task.lr,
                                task = task
                                )
            torch.manual_seed(time.time() * 1000)
            self.y=Net.Mnist_CNN_Net()  
        elif self.model_name == 'Cifar10_CNN_Net':
            torch.manual_seed(42)
            self.model=Model.Models_load(model_name = self.model_name, 
                                server_dataset = self.datasets_test, 
                                model_body = Net.Cifar10_CNN_Net(),  
                                criterion = self.criterion,
                                lambda_l2 = task.lambda_l2,
                                lr = task.lr,
                                task = task
                                )
            torch.manual_seed(time.time() * 1000)
            self.y=Net.Cifar10_CNN_Net()  
        for name, param in self.y.named_parameters():
            param.data = param.data * 0


def create_clients(task = None, 
                   clients_data_index = None, 
                   datasets_train = None, 
                   server_model = None, 
                   y = None):
    clients_list = []
    for i in tqdm(range(task.client_num), desc="Create clients", total=task.client_num):
        client_data_index = clients_data_index[i]
        client_dataset = client_data(datasets_train, client_data_index)
        clients_list.append(
            client(client_data_index = client_data_index, 
                    client_dataset = client_dataset,
                    task = task,
                    rho_k = task.rho[i],
                    eta_k = task.eta[i],
                    loc = task.loc[i],
                    server_model = copy.deepcopy(server_model),
                    y = copy.deepcopy(y)
                    )
                    )
        
    return clients_list


def client_data(datasets_train, client_data_index):
    client_dataset = copy.deepcopy(datasets_train) 
    client_dataset.data = datasets_train.data[client_data_index]
    client_dataset.targets = datasets_train.targets[client_data_index]
    return client_dataset

