from torchvision.transforms import v2
import torch

public_transforms = v2.Compose([
    v2.Resize((256,256)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])