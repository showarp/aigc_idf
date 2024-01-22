import os
from typing import Any
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.augment import *
from PIL import Image


class LoadData(Dataset):
    def __init__(self, root, is_train=True, transforms = []) -> None:
        super().__init__()
        self.transforms = transforms
        root = root
        root_biggan = f"{root}/TinyGenImage/imagenet_ai_0419_biggan"
        root_vqdm = f"{root}/TinyGenImage/imagenet_ai_0419_vqdm"
        root_sdv5 = f"{root}/TinyGenImage/imagenet_ai_0424_sdv5"
        root_wukong = f"{root}/TinyGenImage/imagenet_ai_0424_wukong"
        root_adm = f"{root}/TinyGenImage/imagenet_ai_0508_adm"
        root_glide = f"{root}/TinyGenImage/imagenet_glide"
        root_midjourney = f"{root}/TinyGenImage/imagenet_midjourney"
        root_paths = [root_biggan,root_vqdm,root_sdv5,root_wukong,root_adm,root_glide,root_midjourney]
        if is_train==True:
            root_paths = [f"{path}/train" for path in root_paths]
        else:
            root_paths = [f"{path}/val" for path in root_paths]

        ai_file = []
        nature_file = []
        x_y = []
        for path in root_paths:
            ai_file.append(f"{path}/ai/")
        for path in root_paths:
            nature_file.append(f"{path}/nature/")
        
        for ai in ai_file:
            for ai_img in os.listdir(ai):
                x_y.append((f"{ai}/{ai_img}",1))
        
        for nature in nature_file:
            for nature_img in os.listdir(nature):
                x_y.append((f"{nature}/{nature_img}",0))
        
        self.x_y = x_y
    
    def __len__(self) -> int:
        return len(self.x_y)
    
    def __getitem__(self, index) -> Any:
        x,y = self.x_y[index]
        x = Image.open(x).convert("RGB")
        for transform in self.transforms:
            x = transform(x)
        return x,y

def tiny_genimage_dataloader(root, batch_size=32, num_workers=4):
    """加载dataloader

    Args:
        batch_size (int, optional): batch size. Defaults to 32.
        num_workers (int, optional): num workers. Defaults to 8.

    Returns:
        tupe: train_loader,val_loader
    """
    train_traisnforms = [public_transforms]
    val_traisnforms = [public_transforms]

    train_data = LoadData(root=root, is_train=True, transforms=train_traisnforms)
    val_data = LoadData(root=root, is_train=False, transforms=val_traisnforms)
    train_loader = DataLoader(train_data,batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_data,batch_size, num_workers=num_workers, shuffle=True)
    return train_loader,val_loader