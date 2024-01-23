from loss import CrossEntropyLoss
import torch
import json
import os
from torch.utils.tensorboard import SummaryWriter

num_exp = len(os.listdir('./checkpoint/'))-1
writer = SummaryWriter(f'./checkpoint/exp{num_exp}/log')
def val(net, epoch, loader, device):
    print("Evaluating")
    with open(f'./checkpoint/exp{num_exp}/info.json','r') as f:
        checkpoint_info = json.loads(f.read())
    best_acc = checkpoint_info['best_acc']

    net.eval()
    loss_func = CrossEntropyLoss()
    total_acc = 0
    total_loss = 0
    for idx, (x, y) in enumerate(loader):
        if device != "cpu":
            x, y = x.to(device), y.to(device)
        with torch.no_grad():
            predic = net(x)
            loss = loss_func(predic, y)
        
        batch_size = y.size(0)
        total_acc += torch.argmax(predic,dim=1).eq(y).sum()/batch_size
        total_loss += loss

        print(f"val     epoch:[{epoch}] Iter:{idx:03d}/{len(loader)} Loss:{total_loss/(idx+1):.4f} Acc:{total_acc/(idx+1):.4f}")
        writer.add_scalar('val/Loss', total_loss/(idx+1), global_step=None, walltime=None)
        writer.add_scalar('val/Acc', total_acc/(idx+1), global_step=None, walltime=None)

    if total_acc/len(loader)>best_acc:
        print('Saveing Model...')
        checkpoint_info['best_acc'] = total_acc/len(loader)
        checkpoint_info['epoch'] = epoch
        with open(f'./checkpoint/exp{num_exp}/info.json','w') as f:
            f.write(json.dumps(checkpoint_info))
