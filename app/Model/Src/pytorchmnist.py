# This is the raw code from PytorchMNIST.ipynb
# Is not really used, saved for reference incase someone does not 
# have jupyter notebooks installed
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt

Dataset = MNIST(root="./", transform=torchvision.transforms.ToTensor())
ValSplit = 0.2
TrainSZ, ValSZ = (int(len(Dataset) *( 1-ValSplit )), int(len(Dataset) * ValSplit))

TrainData, ValData = random_split(Dataset, (TrainSZ, ValSZ))

BatchSZ = 100
TrainLoader = DataLoader(TrainData, BatchSZ, shuffle= True)
ValLoader   = DataLoader(ValData, BatchSZ)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ModelStats = namedtuple('ModelStats', ['Loss', 'Accuracy'])

class CNNModel(torch.nn.Module):
    def __init__(self, InputSZ, NClasses):
        super().__init__()
        self.InputSize  = InputSZ
        self.NumClasses = NClasses
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = torch.nn.Linear(3*3*64, 256)
        self.fc2 = torch.nn.Linear(256, NClasses)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return x
    
    def Accuracy(self, Out, lable):
        val, idx = torch.max(Out, dim = 1)
        return torch.tensor(torch.sum(idx == lable).item() / len(idx))
    
    def Step(self, Batch, Validation: bool):
        img, lbl = Batch
        
        img = img.to(device)
        lbl = lbl.to(device)

        out = self(img)
        loss = F.cross_entropy(out, lbl)
        if Validation:
            accuracy = self.Accuracy(out, lbl)
            return ModelStats(loss, accuracy)
        else:
            return loss
    
    def EndValidationEpoch(self, outputs):
        b_loss = [x.Loss for x in outputs]
        b_acc  = [x.Accuracy for x in outputs]

        e_loss = torch.mean(torch.stack(b_loss))
        e_acc  = torch.mean(torch.stack(b_acc))

        return ModelStats(e_loss.item(), e_acc.item())
    
    def EndEpoch(self, e, res):
        print(f"Epoch [{e}] Finished with Loss = {res.Loss:.4}, Accuracy = {(res.Accuracy * 100):.4}%")

def EvaluateModel(model, Loader):
    out = [model.Step(b, True) for b in Loader]
    return model.EndValidationEpoch(out)

def Fit(epochs, model, TrainLoader, ValLoader, opt):
    History = []
    for epoch in range(epochs):
        for batch in TrainLoader:
            loss = model.Step(batch, False)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        res = EvaluateModel(model, ValLoader)
        model.EndEpoch(epoch, res)
        History.append(res)
    
    return History

MODEL = CNNModel(28*28, 10)
MODEL.to(device)

LR = 0.001
MOMENTUM = 0.9
EPOCHS = 15
Optimizer = torch.optim.Adam(MODEL.parameters(), lr= LR)

Hist = Fit(EPOCHS, MODEL, TrainLoader, ValLoader, Optimizer)

Losses = [x.Loss for x in Hist]
Accs   = [x.Accuracy for x in Hist]

plt.plot(Losses, label= "Losses")
plt.plot(Accs, label= "Accuracy")
plt.title("Loss vs Accuracy")
plt.legend()
plt.show()

TestDataset= MNIST(root="./", train=False,transform=torchvision.transforms.ToTensor())

def Predict_Image(img, model, lbl):
    plt.imshow(img[0], cmap="gray")
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    out = model(img)
    _, prediction = torch.max(out, dim=1)
    print(f"Model predicts {prediction[0].item()}, Truth = {lbl}")

img, lbl = TestDataset[9870]
Predict_Image(img, MODEL, lbl)

TestLoader = DataLoader(TestDataset, 10, shuffle=True)
res = EvaluateModel(MODEL, TestLoader)
res

import torch.onnx as onnx

torch.save(MODEL.state_dict(), "CNNModel.pt")

input_image = torch.zeros((1,1,28,28)).to(device)
onnx.export(MODEL, input_image, 'CNNModel.onnx')