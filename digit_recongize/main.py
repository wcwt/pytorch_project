import torch
import torchvision
from torchvision import transforms,datasets

train = datasets.MNIST("",train=True,download=True,
                    transforms = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset  = torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)

for data in trainset:
    print(data)
