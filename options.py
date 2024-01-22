import argparse
from data.data_entry import datas
from model.model_entry import models
from run import run


def parse_args(parser):
    parser.add_argument("--mode", type=str, default="train", help="run mode,default:train")
    parser.add_argument("--model_type", type=str, help="used in model/model_entry.py")
    parser.add_argument("--data_type",type=str,help="used in data/data_entry.py")
    parser.add_argument("--device_type", type=str, default="auto", help="train device, default auto")
    parser.add_argument("--epochs", type=int, default=100, help="epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="train data batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader num of workers")
    parser.add_argument("--num_classes", type=int, help="num of classes")
    return parser


parser = argparse.ArgumentParser()
parser = parse_args(parser)

args = parser.parse_args()
batch_size = args.batch_size
device = args.device_type
lr = args.lr
num_classes = args.num_classes
epochs = args.epochs
num_workers = args.num_workers

model = models[args.model_type]
data = datas[args.data_type]
model = model(num_classes)
data = data(root="./dataset", batch_size=batch_size, num_workers=num_workers)

if __name__ == "__main__":
    run(model, data, epochs, seed=42, device=device, learning_rate=lr)
