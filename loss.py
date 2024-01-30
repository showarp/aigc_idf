from typing import Any
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class SoftAugmentLoss:
    def __init__(self) -> None:
        pass

    def __call__(self,predic,target) -> Any:
        predic = F.softmax(predic,dim=1)
        return F.kl_div(predic.log(),target,reduction="batchmean")
    
class CKLoss:
    def __init__(self) -> None:
        pass
    def __call__(self, predic, target) -> Any:
        """CKLoss

        Args:
            predic (tensor): network output
            target (tensor): soft label

        Returns:
            loss: tensor.float
        """
        predic_softmax = F.log_softmax(predic,dim=1)
        kl_loss = F.kl_div(predic_softmax,target,reduction="batchmean")
        ce_loss = F.cross_entropy(predic,target)
        return kl_loss+ce_loss

# x = torch.tensor([[1., 2., 3.]],dtype=torch.float32)
# y = F.softmax(x,dim=1) 
# x = F.softmax(x,dim=1).log()
# # print(F.softmax(x).log(),F.log_softmax(x))
# print(F.kl_div(x,y,reduction="batchmean"))