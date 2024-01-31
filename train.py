import argparse
from torch import optim
from loss import CrossEntropyLoss,SoftAugmentLoss,CKLoss
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
from data.data_entry import datas
from model.model_entry import models
import os


def train(net, epoch, loader, device, lr, board_writer=None):
    print("Traing")
    net.train()
    optimizer = optim.Adam(params=net.parameters(), lr=lr)
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

        print(
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
        checkpoint = torch.load(weights)
        model = checkpoint["net"].to(device)
        start_epoch = checkpoint["epoch"]
    else:
        model = models[args.model_type]().to(device)
        start_epoch = 0
    data = datas[args.data_type]
    train_loader, _ = data(root="./dataset", batch_size=batch_size, num_workers=num_workers)
    
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    exp_num = len(os.listdir("./checkpoint"))
    os.mkdir(f"./checkpoint/exp{exp_num}")
    writer = SummaryWriter(f"./checkpoint/exp{exp_num}/log")

    for epoch in range(start_epoch, end_epochs):
        train(
            net=model,
            epoch=epoch,
            loader=train_loader,
            device=device,
            lr=lr,
            board_writer=writer,
        )


if __name__ == "__main__":
    main()
