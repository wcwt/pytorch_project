import torch
import torchvision
from torchvision import transforms,datasets

x = torch.Tensor([5,2])
y = torch.zeros([2,5])

z = torch.rand([2,5]) # shape of 2,5
zr = z.view([1,10]) # reshape from 2,5 to 1,10
print(zr)



'''
train = datasets.MNIST("",train=True,download=True,
                    #transforms = transforms.Compose([transforms.ToTensor()])
                    )

trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
#testset  = torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)

for data in trainset:
    print(data[0])
'''
