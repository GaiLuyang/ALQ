import torchvision
from torchvision import datasets, transforms

def dataloading(file_name):
    dictionaries = {}
    if file_name == 'mnist':
        datasets_train = torchvision.datasets.MNIST('mnist data',train=True,download=True,\
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\
        torchvision.transforms.Normalize((0.1307,),(0.3081,))]))

        datasets_test = torchvision.datasets.MNIST('mnist data',train=False,download=True,\
        transform=torchvision.transforms.Compose([\
        torchvision.transforms.ToTensor(),\
        torchvision.transforms.Normalize((0.1307,),(0.3081,))]))
    elif file_name == 'cifar10':
        datasets_train = torchvision.datasets.CIFAR10('cifar10 data', train=True, download=True,  
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
        torchvision.transforms.RandomCrop(32, padding=4),  
        torchvision.transforms.RandomHorizontalFlip(),    
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) 
        ]))

        datasets_test = datasets.CIFAR10('cifar10 data', train=False, download=True, 
        transform = transforms.Compose([torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ]))

    return datasets_train, datasets_test 


