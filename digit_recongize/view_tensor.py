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


for data in trainset:
    print(data)
    break

x,y = data[0][0],data[1][0] # x = image, y = labels,
# x.shape is [1, 28, 28] now

plt.imshow(x.view([28,28]))
plt.show()

print(y)
