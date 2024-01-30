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
from train import train
from val import val


def random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def run(model, loader, seed, device, learning_rate, start_epochs=0, end_epochs=100):
    random_seed(seed)
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    exp_num = len(os.listdir("./checkpoint"))
    os.mkdir(f"./checkpoint/exp{exp_num}")
    info_json = {"best_acc": 0, "epoch": 0}
    info_json = json.dumps(info_json)
    with open(f"./checkpoint/exp{exp_num}/info.json", "w") as f:
        f.write(info_json)
    writer = SummaryWriter(f"./checkpoint/exp{exp_num}/log")

    train_loader, val_loader = loader

    for epoch in range(start_epochs, end_epochs):
        train(
            net=model,
            epoch=epoch,
            loader=train_loader,
            device=device,
            lr=learning_rate,
            board_writer=writer,
        )
        val(
            net=model,
            epoch=epoch,
            loader=val_loader,
            device=device,
            board_writer=writer,
        )


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
    checkpoint = torch.load(weights)
    model = checkpoint["net"].to(device)
    start_epoch = checkpoint["epoch"]
else:
    model = models[args.model_type]().to(device)
    start_epoch = 0
data = datas[args.data_type]
data = data(root="./dataset", batch_size=batch_size, num_workers=num_workers)

if __name__ == "__main__":
    run(
        model=model,
        loader=data,
        start_epochs=start_epoch,
        end_epochs=epochs,
        seed=42,
        device=device,
        learning_rate=lr,
    )
