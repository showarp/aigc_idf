from typing import Any
from torchvision.transforms import v2
import numpy as np
import cv2 as cv
import torch
from PIL import Image


class RandGaussianBlur:
    def __init__(self,p=1.,q=.8) -> None:
        self.p = p
        self.q = int(q*10)

    def __call__(self,img) -> Any:
        if np.random.rand()<self.p:
            linspace = np.linspace(10,0.1,11)
            t = v2.GaussianBlur(kernel_size=9,sigma=linspace[self.q])
            img = t(img)
        return img
    
class JpegZip:
    def __init__(self,p=1.,q=.8) -> None:
        self.p = p
        self.q = int(q*100)

    def __call__(self,img) -> Any:
        if np.random.rand()<self.p:
            img = np.array(img)
            _, img = cv.imencode('.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), self.q])
            img = np.array(img).tobytes()
            img = cv.imdecode(np.frombuffer(img, np.uint8), 1)
            img = Image.fromarray(img)
        return img


public_transforms = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

public_transforms0 = v2.Compose([
    v2.Resize((256,256)),
])

def compose_blur_jpeg(q=.8):
    blur_w = 1*q
    jpeg_w = 1*q
    blur_jpeg_transforms = v2.Compose([
        RandGaussianBlur(q=blur_w),
        JpegZip(q=jpeg_w),
    ])
    return blur_jpeg_transforms

if __name__=="__main__":
    x = Image.open(r'dataset\TinyGenImage\imagenet_ai_0419_biggan\train\nature\n01498041_1932.JPEG').convert("RGB")
    print(123)
    x = public_transforms0(compose_blur_jpeg(.8)(x))
    print(123)
    x.show(x)