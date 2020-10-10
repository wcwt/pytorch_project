import torch
import torchvision
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def plot(img_tensor,predict):
    print(torch.argmax(predict))
    plt.imshow(img_tensor.view([28,28]))
    plt.show()

def test_modle(model,test_case):
    correct = 0
    total = 0
    wrong = 0
    for data in test_case:
        imgs,labels = data[0],data[1]
        output = model(imgs.view(-1,28*28))
        for i , predict in enumerate(output):
            if (torch.argmax(predict) == labels[i]) :
                correct += 1
                if (correct < 5):   plot(imgs[i],predict)
            else:
                wrong += 1
            total += 1
    print(f"Accuracy : {correct*100/total}%")

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

test = train = datasets.MNIST("",train=False, download=False,
                        transform = transforms.Compose([transforms.ToTensor()]))

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

with open("moldel.pk","rb") as f:
    net = pickle.load(f)

test_modle(net,testset)
