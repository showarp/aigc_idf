from torch import optim
from loss import CrossEntropyLoss
import torch
import os
from torch.utils.tensorboard import SummaryWriter


num_exp = len(os.listdir())
writer = SummaryWriter(f'./checkpoint/exp{num_exp}/log')
def train(net, epoch, loader, device, lr):
    print("Traing")
    net.train()
    optimizer = optim.Adam(params=net.parameters(),lr = lr)
    loss_func = CrossEntropyLoss()
    total_acc = 0
    total_loss = 0
    for idx, (x, y) in enumerate(loader):
        if device!="cpu":
            x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        predic = net(x)
        loss = loss_func(predic,y)
        loss.backward()
        optimizer.step()
        
        batch_size = y.size(0)
        total_acc += torch.argmax(predic,dim=1).eq(y).sum()/batch_size
        total_loss += loss

        print(f"train     epoch:[{epoch}] Iter:{idx:03d}/{len(loader)} Loss:{total_loss/(idx+1):.4f} Acc:{total_acc/(idx+1):.4f} Lr:{optimizer.param_groups[0]['lr']}")
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=None, walltime=None)
        writer.add_scalar('train/Loss', total_loss/(idx+1), global_step=None, walltime=None)
        writer.add_scalar('train/Acc', total_acc/(idx+1), global_step=None, walltime=None)


