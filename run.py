import torch
import os
import numpy as np
import random
import json


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


def run(model, loader, epochs, seed, device, learning_rate):
    random_seed(seed)
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    exp_num = len(os.listdir("./checkpoint"))
    os.mkdir(f"./checkpoint/exp{exp_num}")
    info_json = {"best_acc": 0, "epoch": 0}
    info_json = json.dumps(info_json)
    with open(f"./checkpoint/exp{exp_num}/info.json", "w") as f:
        f.write(info_json)

    from train import train
    from val import val

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = loader
    net = model.to(device)

    for epoch in range(epochs):
        train(
            net=net, epoch=epoch, loader=train_loader, device=device, lr=learning_rate
        )
        val(net=net, epoch=epoch, loader=val_loader, device=device)
