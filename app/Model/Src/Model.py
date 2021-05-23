from collections import namedtuple
import torch
import torchvision
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ModelStats = namedtuple('ModelStats', ['Loss', 'Accuracy'])

# Define the model architecture
class CNNModel(torch.nn.Module):
    # TODO: Add typing for parameter
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
    # Computes accuracy by dividing number of correct outputs by number of total outputs 
    def Accuracy(self, Out, lable):
        val, idx = torch.max(Out, dim = 1)
        return torch.tensor(torch.sum(idx == lable).item() / len(idx))
    
    # Helper function run for every training step
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
    # Work out the stats for each epoch
    def EndValidationEpoch(self, outputs):
        b_loss = [x.Loss for x in outputs]
        b_acc  = [x.Accuracy for x in outputs]

        e_loss = torch.mean(torch.stack(b_loss))
        e_acc  = torch.mean(torch.stack(b_acc))

        return ModelStats(e_loss.item(), e_acc.item())
    
    # print details at end of every epoch
    def EndEpoch(self, e, res):
        print(f"Epoch [{e}] Finished with Loss = {res.Loss:.4}, Accuracy = {(res.Accuracy * 100):.4}%")