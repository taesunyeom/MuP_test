import copy, math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb
from torchvision import datasets, transforms
import torchvision
SEED = 42 
torch.manual_seed(SEED)

class FinalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(FinalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.normal_(self.linear.weight, 0, 1)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)/x.shape[-1]

class FinalLinear_2(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(FinalLinear_2, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.normal_(self.linear.weight, 0, 1/math.sqrt(in_features))
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)/math.sqrt(x.shape[-1])

class StdLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(StdLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.normal_(self.linear.weight, 0, 1)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)/math.sqrt(x.shape[-1])

class StdLinear_2(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(StdLinear_2, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.normal_(self.linear.weight, 0, 1/math.sqrt(in_features))
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)

class InputLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(InputLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        init.normal_(self.linear.weight, 0, 1)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x) / math.sqrt(self.in_features)

class InputLinear_2(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(InputLinear_2, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.normal_(self.linear.weight, 0, 1/math.sqrt(in_features))
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)

class MLP_1(nn.Module):
    def __init__(self, D, width=1024, depth=3, gamma0=1.0, num_classes=10, bias=False):
        super(MLP_1, self).__init__()
        self.gamma0 = gamma0
        layers = [InputLinear(D, width, bias=bias), nn.ReLU()]
        for _ in range(depth-2):
            layers.extend([StdLinear(width, width, bias=bias), nn.ReLU()])
        layers.append(FinalLinear(width, num_classes, bias=bias))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.net(x)
        output = x/self.gamma0
        return output

class MLP_2(nn.Module):
    def __init__(self, D, width=1024, depth=3, gamma0=1.0, num_classes=10, bias=False):
        super(MLP_2, self).__init__()
        self.gamma0 = gamma0
        layers = [InputLinear_2(D, width, bias=bias), nn.ReLU()]
        for _ in range(depth-2):
            layers.extend([StdLinear_2(width, width, bias=bias), nn.ReLU()])
        layers.append(FinalLinear_2(width, num_classes, bias=bias))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.net(x)
        output = x/self.gamma0
        return output

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../pytorch-cifar/data', train=True, download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='../pytorch-cifar/data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)
testloader  = DataLoader(testset,  batch_size=100, shuffle=False, num_workers=4)

device = torch.device('cuda')
criterion   = nn.CrossEntropyLoss()

train_loss_dict_1 = []
train_loss_dict_2 = []

WIDTH = 32*32*3
INPUT_DIM = 32*32*3
torch.manual_seed(SEED)
model1 = MLP_1(D=INPUT_DIM, width=WIDTH).double().cuda()
model1_copy = copy.deepcopy(model1)
for param in model1_copy.parameters():
    param.requires_grad = False
torch.manual_seed(SEED)
model2 = MLP_2(D=INPUT_DIM, width=WIDTH).double().cuda()
model2_copy = copy.deepcopy(model2)
for param in model2_copy.parameters():
    param.requires_grad = False

optim1 = optim.SGD(model1.parameters(), lr=0.1*32*32*3)
optim2 = optim.SGD(model2.parameters(), lr=0.1)

for i, (xb, yb) in enumerate(trainloader):
    xb, yb = xb.to(device), yb.to(device)
    xb = xb.double()

    out1 = model1(xb)
    out1_m = model1_copy(xb)
    out2 = model2(xb)
    out2_m = model2_copy(xb)
    
    loss1 = criterion(out1 - out1_m, yb)
    loss2 = criterion(out2 - out2_m, yb)

    train_loss_dict_1.append(loss1.item())
    train_loss_dict_2.append(loss2.item())

    optim1.zero_grad()
    optim2.zero_grad()
    loss1.backward()
    loss2.backward()
    optim1.step()
    optim2.step()
    
    if i % 10 == 0:
        print(f"Step {i}, Loss1 (FRef): {loss1.item():.4f}, Loss2 (SP+OS): {loss2.item():.4f}")


plt.figure(dpi=300)
plt.plot(train_loss_dict_1, label='FRef (MLP_1)')
plt.plot(train_loss_dict_2, label='SP+OS (MLP_2)')
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig('sanitycheck_double.jpg')
plt.close()
