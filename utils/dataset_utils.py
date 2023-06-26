import os
import torch
from utils.dataset_loader.load_apple import load_apple
from utils.dataset_loader.load_diligent import load_diligent
from utils.dataset_loader.load_lightstage import load_lightstage
from utils.dataset_loader.load_diligent100 import load_diligent100
from dataloader import UPSDataset


def build_loader(cfg):
    data_path = os.path.expanduser(cfg.dataset.data_path)
    # Build data loader
    if 'pmsData' in data_path:
        in_data_dict = load_diligent(data_path, cfg)
    elif 'DiLiGenT100' in data_path:
        in_data_dict = load_diligent100(data_path, cfg)
    elif 'LightStage' in data_path:
        in_data_dict = load_lightstage(data_path, scale=1)
    elif 'Apple' in data_path:
        in_data_dict = load_apple(data_path, scale=2)
    else:
        raise NotImplementedError('Unknown dataset')

    obj_n = data_path.split("/")[-1]
    set_n = data_path.split("/")[1]

    batch_size = int(eval(cfg.experiment.batch_size))

    train_set = UPSDataset(
        in_data_dict,
        obj_name=obj_n,
        dataset=set_n,
        gray_scale=cfg.dataset.gray_scale,
        data_len=300,        
    )
    eval_set = UPSDataset(
        in_data_dict,
        obj_name=obj_n,
        dataset=set_n,
        gray_scale=cfg.dataset.gray_scale,
        data_len=300,
    )
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=batch_size, 
                                               num_workers=0,
                                               shuffle=True)
    
    eval_loader  = torch.utils.data.DataLoader(eval_set, 
                                               batch_size=1, 
                                               num_workers=0,
                                               shuffle=False)
    affix = train_set.get_affix()
    return train_loader, eval_loader, affix