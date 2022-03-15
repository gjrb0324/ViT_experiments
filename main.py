import torch
import torchvision
import torchvision.transforms as transforms
import model
import torch.optim as optim

from torchsummary import summary

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,\
                                        download=True)
testset = torchvision.testsets.CIFAR10(root='./data', trian=False,\
                                       download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,\
                                          shuffle = True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4,\
                                         shuffle=False, num_workers = 0)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship',\
           'truck')

net = model.ViT()

criterion = nn.CrossEntropyLoss()
optimizr = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data


embedding = model.Embedding()
embedded = embedding(img)
tran_enc = model.TransformerEncoderLayer(8)(embedded)
print(tran_enc.size())
