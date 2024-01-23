from .tiny_genimage import tiny_genimage_dataloader
from .tiny_sdv5 import tiny_sdv5_dataloader

datas = {
    "tiny_genimage":tiny_genimage_dataloader,
    "tiny_sdv5":tiny_sdv5_dataloader,
}