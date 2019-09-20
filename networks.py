
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
class RegressionNet(nn.Module):
    def __init__(self, n_outputs):
        super(RegressionNet, self).__init__()

        # Use alexnet as a pretrained model. We want to adapt this
        # so that we can choose multiple pretrained nets
        self.model = torchvision.models.alexnet(pretrained=True)

        # Freeze alexnet
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Custom network appended to pretrained network. 
        num_ftrs =self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, 1000),
                            nn.ReLU(),
                            nn.Dropout(),
                            nn.Linear(1000, 500),
                            nn.ReLU(),
                            nn.Linear(500, 100),
                            nn.ReLU(),
                            nn.Linear(100, n_outputs)) 
 
  

    def forward(self, x):
        return self.model(x)

#Unets will go here