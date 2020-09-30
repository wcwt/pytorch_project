import torch
import torchvision
from torchvision import transforms,datasets
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST("",train=True, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

test = train = datasets.MNIST("",train=False, download=False,
                        transform = transforms.Compose([transforms.ToTensor()]))

# batch_size ==> 10 sample at a time per model
# lesses batches ==>less sample at a time > more optimizes times will do and the more general rule will be generated !!!
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):

    def __init__(self):
        super().__init__()
