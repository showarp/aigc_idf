from .tiny_genimage import tiny_genimage_dataloader
from .tiny_sdv5 import tiny_sdv5_dataloader
from .tiny_wukong import tiny_wukong_dataloader
from .tiny_adm import tiny_adm_dataloader
from .tiny_biggan import tiny_biggan_dataloader
from .tiny_glide import tiny_glide_dataloader
from .tiny_mj import tiny_mj_dataloader
from .tiny_vqdm import tiny_vqdm_dataloader
from .tiny_genimage_soft import tiny_genimage_soft_dataloader
from .tiny_sdv5_soft import tiny_sdv5_soft_dataloader
from .tiny_wukong_soft import tiny_wukong_soft_dataloader
from .tiny_adm_soft import tiny_adm_soft_dataloader
from .tiny_biggan_soft import tiny_biggan_soft_dataloader
from .tiny_glide_soft import tiny_glide_soft_dataloader
from .tiny_mj_soft import tiny_mj_soft_dataloader
from .tiny_vqdm_soft import tiny_vqdm_soft_dataloader

datas = {
    "tiny_genimage":tiny_genimage_dataloader,
    "tiny_sdv5":tiny_sdv5_dataloader,
    "tiny_wukong":tiny_wukong_dataloader,
    "tiny_adm":tiny_adm_dataloader,
    "tiny_biggan":tiny_biggan_dataloader,
    "tiny_glide":tiny_glide_dataloader,
    "tiny_mj":tiny_mj_dataloader,
    "tiny_vqdm":tiny_vqdm_dataloader,
    "tiny_genimage_soft":tiny_genimage_soft_dataloader,
    "tiny_sdv5_soft":tiny_sdv5_soft_dataloader,
    "tiny_wukong_soft":tiny_wukong_soft_dataloader,
    "tiny_adm_soft":tiny_adm_soft_dataloader,
    "tiny_biggan_soft":tiny_biggan_soft_dataloader,
    "tiny_glide_soft":tiny_glide_soft_dataloader,
    "tiny_mj_soft":tiny_mj_soft_dataloader,
    "tiny_vqdm_soft":tiny_vqdm_soft_dataloader,
}