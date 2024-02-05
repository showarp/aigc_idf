from loss import CKLoss, CrossEntropyLoss, SoftAugmentLoss
import torch
import json
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
from data.data_entry import datas
from utlise import log_print
import os


def val(net, epoch, loader, device, is_triang=True, board_writer=None):
    if is_triang:
        num_exp = len(os.listdir("./checkpoint/")) - 1
        with open(f"./checkpoint/exp{num_exp}/info.json", "r") as f:
            checkpoint_info = json.loads(f.read())
            best_acc = checkpoint_info["best_acc"]
        lprint = log_print(f"./checkpoint/exp{num_exp}/runing_log.txt")
    else:
        lprint = log_print(save_log=False) 

    print("Evaluating")
    net.eval()
    # loss_func = CrossEntropyLoss()
    loss_func = SoftAugmentLoss()
    # loss_func = CKLoss()
    total_acc = 0
    total_loss = 0
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            predic = net(x)
            loss = loss_func(predic, y)

        batch_size = y.size(0)
        # total_acc += torch.argmax(predic, dim=1).eq(y).sum() / batch_size
        total_acc += torch.argmax(predic, dim=1).eq(torch.argmax(y, dim=1)).sum() / batch_size
        total_loss += loss

        lprint(f"val     epoch:[{epoch}] Iter:{idx:03d}/{len(loader)} Loss:{total_loss/(idx+1):.4f} Acc:{total_acc/(idx+1):.4f}")
    
    if is_triang and board_writer:
        board_writer.add_scalar("val/Loss", total_loss / len(loader), global_step=epoch, walltime=None)
        board_writer.add_scalar("val/Acc", total_acc / len(loader), global_step=epoch, walltime=None)

    if is_triang and total_acc / len(loader) > best_acc:
        lprint("Saveing Model...")
        checkpoint_info["best_acc"] = float(total_acc / len(loader))
        checkpoint_info["epoch"] = int(epoch)
        with open(f"./checkpoint/exp{num_exp}/info.json", "w") as f:
            f.write(json.dumps(checkpoint_info))
        save_content = {
            "net": net,
            "acc": float(total_acc / len(loader)),
            "epoch": epoch,
        }
        if not os.path.exists(f"./checkpoint/exp{num_exp}/models/"):
            os.mkdir(f"./checkpoint/exp{num_exp}/models/")
        torch.save(save_content, f"./checkpoint/exp{num_exp}/models/best.pth")


def parse_args(parser):
    parser.add_argument("--weights", default="", type=str, help="model weight")
    parser.add_argument("--data_type", type=str, help="used in data/data_entry.py")
    parser.add_argument("--device_type", type=str, default="auto", help="train device, default auto")
    parser.add_argument("--batch_size", type=int, default=32, help="train data batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader num of workers")
    return parser


def main():
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")

    parser = argparse.ArgumentParser()
    args = parse_args(parser).parse_args()

    batch_size    = args.batch_size
    device        = args.device_type
    num_workers   = args.num_workers
    weights       = args.weights
    
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if os.path.exists(weights):
        checkpoint = torch.load(weights,map_location=device)
        model = checkpoint["net"].to(device)
    data = datas[args.data_type]
    _, val_loader = data(root="./dataset", batch_size=batch_size, num_workers=num_workers)

    val(net=model, epoch=0, loader=val_loader, device=device, is_triang=False)


if __name__ == "__main__":
    main()
