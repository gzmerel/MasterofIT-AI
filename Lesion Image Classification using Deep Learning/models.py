import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class SimpleBNConv(nn.Module):
  def __init__(self,num_classes = 7):
    super().__init__()
    self.feature_extractor = nn.Sequential(
      #First Layer
      nn.Conv2d(3,8,3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(8),
      nn.MaxPool2d(2,2),
      #Second Layer 
      nn.Conv2d(8,16,3,padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.MaxPool2d(2,2),
      #Third Layer  
      nn.Conv2d(16,32,3,padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(2,2),
      #Fourth Layer 
      nn.Conv2d(32,64,3,padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2,2),
      #Fifth Layer 
      nn.Conv2d(64,128,3,padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.MaxPool2d(2,2),
    )
    self.classifier = nn.Sequential(
      nn.AdaptiveAvgPool2d(1) 
      nn.Flatten(),
      #nn.LazyLinear(num_classes)
      nn.Linear(128, num_classes)
    )
    #Tranfer the model weights to the GPU
  def forward(self,x):
    if x.ndim == 5:
      B, N, C, H, W = x.shape
      x=x.view(B*N,C, H, W)
      x=self.feature_extractor(x)
      x=self.classifier(x)
      x= x.view(B,N,-1).mean(dim=1)
    else:
      x = self.feature_extractor(x)
      x = self.classifier(x)
    return x
# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.

class PretrainedResNet18(nn.Module):
  def __init__(self, num_classes= 7, freeze = True):
    super().__init__()
    #Loading the pretrained ResNetModel
    self.base_model = models.resnet18(pretrained = True)
    
    #Replace the final fully connected Layer
    in_features = self.base_model.fc.in_features
    self.base_model.fc = nn.Linear(in_features , num_classes)

  def forward(self,x):
    return self.base_model(x)

# TODO Task 1f - Create your own models
class CustomCNNv1(nn.Module):
  def __init__(self,num_classes = 7):
    super().__init__()
    self.feature_extractor = nn.Sequential(
      #First Layer
      nn.Conv2d(3,16,3, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.MaxPool2d(2,2),
      #Second Layer 
      nn.Conv2d(16,32,3,padding = 1),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(2,2),
      #Third Layer  
      nn.Conv2d(32,64,3,padding = 1),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2,2),
      #Fourth Layer
      nn.Conv2d(64,128,3,padding = 1),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.BatchNorm2d(128),
      nn.MaxPool2d(2,2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(num_classes)
    )
    #Tranfer the model weights to the GPU
  def forward(self,x):
    if x.ndim == 5:
      B, N, C, H, W = x.shape
      x=x.view(B*N,C, H, W)
      x=self.feature_extractor(x)
      x=self.classifier(x)
      x= x.view(B,N,-1).mean(dim=1)
    else:
      x = self.feature_extractor(x)
      x = self.classifier(x)
    return x




































