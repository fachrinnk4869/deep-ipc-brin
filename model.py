from xr14 import xr14
from config import GlobalConfig as Config
from vit_bb_ss import vit_bb_ss
from eff_vit import eff_vit
from setr_x14_x14 import setr_x14_x14

config = Config()


def get_model(*args, **kwargs):
    model_name = config.model
    if model_name == 'vit_bb':
        return vit_bb_ss(*args, **kwargs)
    elif model_name == 'xr14':
        return xr14(*args, **kwargs)
    elif model_name == 'eff_vit':
        return eff_vit(*args, **kwargs)
    elif model_name == 'setr_x14_x14':
        return setr_x14_x14(*args, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
