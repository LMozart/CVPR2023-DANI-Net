import torch
import numpy as np
from tqdm import trange
from utils.utils import *
from utils.dataset_utils import build_loader
from ops.config_parser import ExpConfig
from utils.record_utils import ExpLog
from models.tester import DepthBaseTester

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
    cfg.logger_type = "None"
    logger = ExpLog(args, cfg)
    cfg.models.checkpoint_path = logger.log_path

    # Set up dataloader.
    _, eval_loader, affix = build_loader(cfg)

    # Set up runner.
    runner = DepthBaseTester(cfg, device, affix)
    runner.load_checkpoints(cfg)
    runner.set_model_status("eval")
    with torch.no_grad():
        for _, eval_data in enumerate(eval_loader, start=1):
            eval_pred = runner.model_pred(eval_data, 0, 1)
            metrics = runner.eval(eval_pred, eval_data)
            logger.plot_lighting(eval_pred["est_ldir"].detach().cpu().numpy(), eval_pred["est_lint"].detach().cpu().numpy())
            break
    print(f"[Test Over]: lgt mae - {metrics['lgt_mae']}, lgt int - {metrics['lgt_int']}, nml mae - {metrics['nml_mae']}")
    logger.log_status(0, metrics, "test")