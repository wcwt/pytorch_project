import torch
import torchvision
from torchvision import transforms,datasets
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

train = datasets.MNIST("",train=True, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

test = train = datasets.MNIST("",train=False, download=False,
                        transform = transforms.Compose([transforms.ToTensor()]))

# batch_size ==> 10 sample at a time per model
# lesses batches ==>less sample at a time > more optimizes times will do and the more general rule will be generated !!!
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

ls = [28*28, 64, 64, 64, 10] # layer_size
num_of_layer = len(ls) - 1

class Net(nn.Module):
    def __init__(self):
        self.fc = [] # fully connected layer
        super().__init__() # the same as nn.Module.__init__()
        for i in range(num_of_layer):
            self.fc.append( nn.Linear(ls[i], ls[i+1]) ) # input,output # get the seeting of the layer

    def forward(self,x):
        # set activation function in the middle layer
        for i in range(num_of_layer - 1 ):
            x = F.relu( self.fc[i](x) )
        x = self.fc[-1](x) # the last layer no need any activation function
        return F.log_softmax(x, dim=1) # (Qn) I don't understand what dim one is
net = Net()
#net( torch.rand([28*28]).view(-1,28*28) )
#optimizer  = optim.Adam(net.parameters(), lr=0.001) # para that do not adjust

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data contains image,result
        imgs,labels = data[0],data[1]
        net.zero_grad()
