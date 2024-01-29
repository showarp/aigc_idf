from typing import Any
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class SoftAugmentLoss:
    def __init__(self) -> None:
        pass

    def __call__(self,predic,target) -> Any:
        predic = F.log_softmax(predic,dim=1)
        return F.kl_div(predic,target,reduction="batchmean")