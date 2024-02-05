import torch
import os
import numpy as np
import random
import json
import argparse
from data.data_entry import datas
from model.model_entry import models
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch import optim
from train import train
from val import val
from utlise import create_checkpoint


def random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def run(log_info,model, loader, seed, device, start_epochs=0, end_epochs=100):
    random_seed(seed)
    log_file = create_checkpoint()
    info_json = {"best_acc": 0, "epoch": 0}
    info_json = {**info_json,**log_info}
    info_json = json.dumps(info_json)
    with open(f"{log_file}/info.json", "w") as f:
        f.write(info_json)
    writer = SummaryWriter(f"{log_file}/log")

    train_loader, val_loader = loader
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(start_epochs, end_epochs):
        train(
            net=model,
            epoch=epoch,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            board_writer=writer,
        )
        val(
            net=model,
            epoch=epoch,
            loader=val_loader,
            device=device,
            board_writer=writer,
        )
        scheduler.step()


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


if not os.path.exists("./dataset"):
    os.mkdir("./dataset")

parser = argparse.ArgumentParser()
args = parse_args(parser).parse_args()

batch_size      = args.batch_size
device          = args.device_type
lr              = args.lr
epochs          = args.epochs
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
data = data(root="./dataset", batch_size=batch_size, num_workers=num_workers)

if __name__ == "__main__":
    run(
        log_info = {"data_type":args.data_type,"model_type":args.model_type},
        model=model,
        loader=data,
        start_epochs=start_epoch,
        end_epochs=epochs,
        seed=42,
        device=device,
    )
