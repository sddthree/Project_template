import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

class MIL_fc(nn.Module):
    def __init__(self, config=None, n_classes=2):
        super(MIL_fc, self).__init__()

        self.fc = nn.Sequential(*[nn.Linear(384, 1), nn.ReLU()])

        self.classifier= nn.Linear(384, n_classes)
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, coad):
        # print(coad.shape, type(coad))
        h = self.fc(coad)
        coad = coad.transpose(1, 2)
        coad = torch.matmul(coad, h)
        coad = coad.transpose(1, 2).squeeze(0)
        # print(coad.shape)
        logits = self.classifier(coad)
    
        return logits


