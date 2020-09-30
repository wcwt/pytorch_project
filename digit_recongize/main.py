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

class Net(nn.Module):
    def __init__(self):
        self.fc = [] # fully connected layer
        super().__init__() # the same as nn.Module.__init__()
        self.fc1 = nn.Linear(ls[0], ls[1])
        self.fc2 = nn.Linear(ls[1], ls[2])
        self.fc3 = nn.Linear(ls[2], ls[3])
        self.fc4 = nn.Linear(ls[3], ls[4])


    def forward(self,x):
        # set activation function in the middle layer
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = self.fc4(x)  # the last layer no need any activation function
        return F.log_softmax(x, dim=1) # (Qn) I don't understand what dim one is

net = Net()
print(net)
#net( torch.rand([28*28]).view(-1,28*28) )
optimizer  = optim.Adam(net.parameters(), lr=0.001) # para that do not adjust

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data contains image,result
        imgs,labels = data[0],data[1]
        net.zero_grad()
        output = net(imgs.view(-1,28*28))
        # error handling
        loss = F.nll_loss(output,labels)
        loss.backward() # magical
        optimizer.step()
    print(loss)
