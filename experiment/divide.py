import torch
import time

def onehot(label,depth=10):
    out=torch.zeros(label.size(0),depth)
    idx=torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out

def dirichlet_split_noniid(datasets_train, datasets_test, task):
    if task.choose == 1:
        datasets_train.targets = torch.tensor(datasets_train.targets)
        datasets_test.targets = torch.tensor(datasets_test.targets)
    n_classes = len(torch.unique(datasets_train.targets))
    torch.manual_seed(1933)
    label_distribution = torch.distributions.Dirichlet(torch.full((task.client_num,), task.noniid_level)).sample((n_classes,))  
    torch.manual_seed(time.time()*1000)
    # 1. Get the index of each label
    labels = datasets_train.targets.unsqueeze(1).clone().detach().to(torch.float32)
    class_idcs = [torch.nonzero(labels == y)[:, 0]  for y in range(n_classes)]    
    # 2. According to the distribution, the label is assigned to each client
    client_idcs = [[] for _ in range(task.client_num)]    
    for c, fracs in zip(class_idcs, label_distribution):
        total_size = len(c)
        splits = (fracs * total_size).int()   
        splits[-1] = total_size - splits[:-1].sum()  
        idcs = torch.split(c, splits.tolist())  
        for i, idx in enumerate(idcs):
            client_idcs[i] += [idcs[i]]

    client_idcs = [torch.cat(idcs) for idcs in client_idcs]
    for i in range(len(client_idcs)):
        if len(client_idcs[i])< task.batch_size:
            print('Samples in client {} are less than batchsize, put {} samples from server to client {}'.format(i, task.batch_size-len(client_idcs[i]), i))
            client_idcs[i], datasets_train, datasets_test = sample_complement(client_idcs[i], task.batch_size, datasets_train, datasets_test)
    return client_idcs, datasets_train, datasets_test 



def sample_complement(client_idcs, batch_size, datasets_train, datasets_test):
    while len(client_idcs)<batch_size:
        datasets_test, sample, target = remove_one_data(datasets_test)
        datasets_train = add_one_data(datasets_train, sample, target)
        train_datanum = datasets_train.data.size()[0]
        client_idcs = torch.cat((client_idcs, torch.tensor([train_datanum-1])))
    return client_idcs, datasets_train, datasets_test

def remove_one_data(datasets_test):
    torch.manual_seed(torch.initial_seed() % (2**32 - 1))
    random_int = torch.randint(0, len(datasets_test.data), (1,)).item()
    sample = datasets_test.data[random_int]
    target = datasets_test.targets[random_int]
    datasets_test.data = torch.cat((datasets_test.data[:random_int], datasets_test.data[random_int+1:]), dim=0)
    datasets_test.targets = torch.cat((datasets_test.targets[:random_int], datasets_test.targets[random_int+1:]), dim=0)
    return datasets_test, sample, target

def add_one_data(datasets_train, sample, target):
    datasets_train.data = torch.cat((datasets_train.data, torch.unsqueeze(sample, 0)), dim=0)
    datasets_train.targets = torch.cat((datasets_train.targets, torch.unsqueeze(target, 0)), dim=0)
    return datasets_train





