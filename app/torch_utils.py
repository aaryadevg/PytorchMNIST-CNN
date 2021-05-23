import io # Reading image as a stream 
import torch # deep learning
import torch.nn as nn # Model definition
import torchvision.transforms as Transform # Transform data before sending into the model
from PIL import Image # Image IO
import torch.nn.functional as F # Activation functions and loss functions
from Model.Src.Model import CNNModel # Loading pretrained model created in google collab

InputSz  = 784
NClasses = 10

# TODO: Use OS.path
PATH = "Model\Saves\CNNModel.pt" # This currently needs changes when switching between OS

model = CNNModel(InputSz, NClasses)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'))) # Loading model weights
model.eval() # Set model into evaluation mode

def TransformImg(imgBytes):
    transform = Transform.Compose([
        Transform.Grayscale(num_output_channels= 1),
        Transform.Resize((28,28)), # This allows the user to send any size images
        Transform.ToTensor()
    ])
    
    image = Image.open(io.BytesIO(imgBytes))
    # Unsqueeze adds a dimension, since torch expects a batch of inputs
    # instead of 1 input
    return transform(image).unsqueeze(0)

def GetPrediction(ImageTensor):
    Output      = model(ImageTensor)
    _, Prediction = torch.max(Output, 1) # Discard the probabilities
    return Prediction