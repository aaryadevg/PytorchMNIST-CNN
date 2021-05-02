import io
import torch
import torch.nn as nn
import torchvision.transforms as Transform
from PIL import Image
import torch.nn.functional as F
from Model.Src.Model import CNNModel

InputSz  = 784
NClasses = 10
PATH = "Model\Saves\CNNModel.pt"

model = CNNModel(InputSz, NClasses)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

def TransformImg(imgBytes):
    transform = Transform.Compose([
        Transform.Grayscale(num_output_channels= 1),
        Transform.Resize((28,28)),
        Transform.ToTensor()
    ])
    
    image = Image.open(io.BytesIO(imgBytes))
    return transform(image).unsqueeze(0)

def GetPrediction(ImageTensor):
    Output      = model(ImageTensor)
    _, Prediction = torch.max(Output, 1)
    return Prediction