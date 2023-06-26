import torch
import numpy as np
from tqdm import trange
from utils.utils import *
from utils.dataset_utils import build_loader
from ops.config_parser import ExpConfig
from utils.record_utils import ExpLog
from models.runner import DepthBaseRunner

if __name__ == '__main__':
    args, cfg = ExpConfig().parse()

    # Set random numbers.
    if cfg.experiment.randomseed is not None:
        np.random.seed(cfg.experiment.randomseed)
        torch.manual_seed(cfg.experiment.randomseed)
        torch.cuda.manual_seed_all(cfg.experiment.randomseed)

    # Set devices.
    if args.cuda is not None:
        cfg.experiment.cuda = f"cuda:{args.cuda}"
    device = torch.device(cfg.experiment.cuda)

    # Set up logger.
    logger = ExpLog(args, cfg)

    # Set up dataloader.
    train_loader, eval_loader, affix = build_loader(cfg)

    # Set up runner.
    runner = DepthBaseRunner(cfg, device, affix)

    start_epoch = cfg.experiment.start_epoch
    end_epoch   = cfg.experiment.end_epoch

    iter_count = start_epoch * len(train_loader) + 1

    with trange(start_epoch, end_epoch + 1) as pbar:
        for epoch in pbar:
            for iter_num, in_data in enumerate(train_loader):
                pred = runner.model_pred(in_data, epoch, end_epoch)
                metrics = runner.train(cfg, pred, in_data, end_epoch, epoch)
                logger.log_status(iter_count, metrics, "train")
                
                nml_mae = metrics["nml_mae"]
                lgt_mae = metrics["lgt_mae"]
                pbar.set_description(f"nml mae:{nml_mae:.2f}, lgt mae:{lgt_mae:.2f}")

                iter_count += 1
            
            runner.scheduler.step()

            if epoch % cfg.experiment.eval_every_iter == 0:
                runner.set_model_status("eval")
                with torch.no_grad():
                    for _, eval_data in enumerate(eval_loader, start=1):
                        eval_pred = runner.model_pred(eval_data, epoch, end_epoch)
                        metrics = runner.eval(eval_pred, eval_data)
                        break
                runner.set_model_status("train")
                logger.log_status(iter_count, metrics, "eval")
                
            if epoch % cfg.experiment.save_every_epoch == 0:
                ckpt = runner.fetch_checkpoints(epoch)
                logger.save_checkpoints(ckpt, epoch)
