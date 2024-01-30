import os
from typing import Any
from torch.utils.data import Dataset, DataLoader
from .augment import *
from PIL import Image


class LoadData(Dataset):
    def __init__(self, root, is_train=True, transforms = []) -> None:
        super().__init__()
        self.transforms = transforms
        root = root
        root_sdv5 = f"{root}/TinyGenImage/imagenet_ai_0424_sdv5"
        if is_train==True:
            root_paths = f"{root_sdv5}/train"
        else:
            root_paths = f"{root_sdv5}/val"

        ai_file = f"{root_paths}/ai"
        nature_file = f"{root_paths}/nature"
        x_y = []
        
        for ai_img in os.listdir(ai_file):
            x_y.append((f"{ai_file}/{ai_img}",1))
        
        for nature_img in os.listdir(nature_file):
            x_y.append((f"{nature_file}/{nature_img}",0))
        
        self.x_y = x_y
    
    def __len__(self) -> int:
        return len(self.x_y)
    
    def __getitem__(self, index) -> Any:
        x,y = self.x_y[index]
        x = Image.open(x).convert("RGB")
        img_quality = 1
        n_classes = 2
        for transform in self.transforms:
            if transform == "blur":
                img_quality = np.random.randint(100,101)*0.01
                t = compose_blur_jpeg(img_quality,p=.3)
                x = t(x)
                continue
            x = transform(x)
        p = torch.tensor(1-(1-1/2)*(1-img_quality)**2)
        soft_label = torch.zeros(2)
        soft_label[::] = 1-p/(n_classes-1)
        soft_label[y] = p
        print(soft_label)
        return x,soft_label

def tiny_sdv5_soft_dataloader(root, batch_size=32, num_workers=4):
    """加载dataloader

    Args:
        batch_size (int, optional): batch size. Defaults to 32.
        num_workers (int, optional): num workers. Defaults to 8.

    Returns:
        tupe: train_loader,val_loader
    """
    train_traisnforms = [public_transforms0,"blur",public_transforms]
    val_traisnforms = [public_transforms0,public_transforms]

    train_data = LoadData(root=root, is_train=True, transforms=train_traisnforms)
    val_data = LoadData(root=root, is_train=False, transforms=val_traisnforms)
    train_loader = DataLoader(train_data,batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data,batch_size, num_workers=num_workers, shuffle=True)
    return train_loader,val_loader
