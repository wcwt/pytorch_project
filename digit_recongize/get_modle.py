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

test = train = datasets.MNIST("",train=False, download=False,
                        transform = transforms.Compose([transforms.ToTensor()]))

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

with open("moldel.pk","rb") as f:
    net = pickle.load(f)

#test_modle(net,testsetB)
