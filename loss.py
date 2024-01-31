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
        predic_softmax = F.softmax(predic,dim=1)
        target_softmax = F.softmax(target,dim=1)
        kl_loss = F.kl_div(predic_softmax.log(),target_softmax,reduction="batchmean")
        target = target.argmax(dim=1)
        # target[::] = 0
        # target = target.scatter(1,target_max,1)
        ce_loss = F.cross_entropy(predic,target)
        return kl_loss+ce_loss

# lossfun = CKLoss()
# x = torch.tensor([[.3,.3,.4]])
# y = torch.tensor([[.3,.3,.4],[.3,.3,.4]])
# print(lossfun(x,y))