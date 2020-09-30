import torch
import torchvision
from torchvision import transforms,datasets
import matplotlib.pyplot as plt

train = datasets.MNIST("",train=True, download=True,
                        transform = transforms.Compose([transforms.ToTensor()]))

test = train = datasets.MNIST("",train=False, download=False,
                        transform = transforms.Compose([transforms.ToTensor()]))

# batch_size ==> 10 sample at a time per model
# lesses batches ==>less sample at a time > more optimizes times will do and the more general rule will be generated !!!
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

count = torch.zeros([10])


for data in trainset:
    for label in data[1]:
        count[int(label)] += 1

sum = sum(count)
for i in range(len(count)):
    print(f"{i}: {100*count[i]/sum}")
