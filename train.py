import argparse
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from loss import CrossEntropyLoss,SoftAugmentLoss,CKLoss
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
from data.data_entry import datas
from model.model_entry import models
import os
from utlise import log_print,create_checkpoint


def train(net, epoch, loader, device, optimizer,only_train=False,board_writer=None):
    if not only_train:
        num_exp = len(os.listdir("./checkpoint/")) - 1
        lprint = log_print(f"./checkpoint/exp{num_exp}/runing_log.txt")
    else:
        lprint = log_print(save_log=False)
    lprint("Traing")
    net.train()
    
    # loss_func = CrossEntropyLoss()
    loss_func = SoftAugmentLoss()
    # loss_func = CKLoss()
    total_acc = 0
    total_loss = 0
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        predic = net(x)
        loss = loss_func(predic, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        # total_acc += torch.argmax(predic, dim=1).eq(y).sum() / batch_size
        total_acc += torch.argmax(predic, dim=1).eq(torch.argmax(y, dim=1)).sum() / batch_size

        total_loss += loss

        lprint(
            f"train     epoch:[{epoch}] Iter:{idx:03d}/{len(loader)} Loss:{total_loss/(idx+1):.4f} Acc:{total_acc/(idx+1):.4f} Lr:{optimizer.param_groups[0]['lr']}",
            end="\r"
        )
        if board_writer:
            board_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=(epoch*len(loader)+idx), walltime=None)
            board_writer.add_scalar("train/Loss", total_loss / (idx + 1), global_step=(epoch*len(loader)+idx), walltime=None)
            board_writer.add_scalar("train/Acc", total_acc / (idx + 1), global_step=(epoch*len(loader)+idx), walltime=None)


def parse_args(parser):
    parser.add_argument("--model_type", type=str, help="used in model/model_entry.py")
    parser.add_argument("--weights", default="", type=str, help="model weight")
    parser.add_argument("--data_type", type=str, help="used in data/data_entry.py")
    parser.add_argument("--device_type", type=str, default="auto", help="train device, default auto")
    parser.add_argument("--epochs", type=int, default=100, help="epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="train data batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader num of workers")
    return parser


def main():
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")

    parser = argparse.ArgumentParser()
    args = parse_args(parser).parse_args()

    batch_size      = args.batch_size
    device          = args.device_type
    lr              = args.lr
    end_epochs      = args.epochs
    num_workers     = args.num_workers
    weights         = args.weights
    
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if os.path.exists(weights):
        checkpoint = torch.load(weights,map_location=device)
        model = checkpoint["net"].to(device)
        start_epoch = checkpoint["epoch"]
    else:
        model = models[args.model_type]().to(device)
        start_epoch = 0
    data = datas[args.data_type]
    train_loader, _ = data(root="./dataset", batch_size=batch_size, num_workers=num_workers)
    
    log_file = create_checkpoint()
    writer = SummaryWriter(f"{log_file}/log")

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(start_epoch, end_epochs):
        train(
            net=model,
            epoch=epoch,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            board_writer=writer,
        )


if __name__ == "__main__":
    main()
