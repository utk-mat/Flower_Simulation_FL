import torch

from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

def get_mnist(data_path: str = './data'):
    
    tr = Compose([ToTensor(),Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, download=True, transform=tr)
    testset = MNIST(data_path , train=False, download=True, transform=tr)

    return trainset, testset




def prepare_dataset(num_partitions: int,
                    batch_size: int,
                    val_ratio: float=0.1):

    trainset,testset = get_mnist()

    # split trainset into 'num_partitions' trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset,partition_len, torch.Generator().manual_seed(2023) )

    #create dataloaders with train+val supoort
    trainloader = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        
        for_train, for_val = random_split(trainset_,[num_train, num_val],torch.Generator().manual_seed(2023))

        trainloader.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=False, num_workers=2))


    testloader = DataLoader(testset, batch_size=128)

    return trainloader, valloaders, testloader