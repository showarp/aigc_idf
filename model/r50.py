import torch.nn as nn
import torch.functional as F
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, num_class=2) -> None:
        super().__init__()
        net = resnet50()
        num_in_features = net.fc.in_features
        net.fc = nn.Linear(
            in_features=num_in_features, out_features=num_class, bias=True
        )
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return y