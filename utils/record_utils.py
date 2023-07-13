import os
import wandb
import cv2 as cv
import matplotlib.pyplot as plt
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np
import imageio

def process_images(pixels, h, w, channel, idxp, idxp_invalid, bounding_box_int):
    img = torch.ones((h, w, channel), device=pixels.device)
    img[idxp] = pixels
    img = img.cpu().numpy()
    img = np.clip(img * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
    img[idxp_invalid] = 255
    img = img[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
    return img

def set_figscale(fig, ax):
    x0, y0, dx, dy = ax.get_position().bounds
    w = 3 * max(dx, dy) /dx
    h = 3 * max(dx, dy) /dy
    fig.set_size_inches((w, h))

def draw_circle(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    x, y = np.cos(t), np.sin(t)
    ax.plot(x*1.0, y*1.0, 'k')
    axis = 1.01
    ax.axis([-axis, axis, -axis, axis])

class ExpLog(object):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg  = cfg
        self.setup_log()
    
    def _plot_light(self, x, y, save_name, c=None):
        # If the input is from MLP network, need to convert y = -y
        y = -y
        fig, ax = plt.subplots()

        if c is None:
            ax.scatter(x, y, s=6)
        else:
            plt.scatter(x, y, c=c, cmap='jet', vmin=0, vmax=1)
        draw_circle(ax)

        ax.axis('off')
        set_figscale(fig, ax)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(save_name, bbox_inches=extent)
        plt.close()

    def plot_lighting(self, dirs, ints):
        # Visualize light direction and intensity
        save_name = os.path.join(self.log_path_img, 'est_light_map.png')
        if len(ints.shape) > 1:
            ints = ints.mean(-1)
        ints = ints / ints.max()
        self._plot_light(dirs[:,0], dirs[:, 1], save_name, ints)

    def make_log_path(self):
        log_path_spl     = self.cfg.experiment.log_path.split("/")
        log_path_spl[-2] += f"_{self.args.exp_code}"
        log_path = "/".join(log_path_spl)
        print(f"[INFO] Log Paths:{log_path}")
        self.log_path  = os.path.expanduser(log_path)

    def setup_log(self):
        self.make_log_path()
        if self.cfg.logger_type == "wandb":
            wandb.init(project="CVPR-Final",
                       config=self.cfg,
                       tags=[self.args.exp_code])
            obj_name = self.args.config.split("/")[-1].split(".")[0]
            wandb.run.name = f"{obj_name}_ {wandb.run.id}"
            wandb.save(self.args.config)
            wandb.run.log_code(".")
        elif self.cfg.logger_type == "tensorboard":
            self.writer = SummaryWriter(self.log_path)
        else:
            self.writer = None

        # Make up dirs    
        self.log_path_img = pjoin(self.log_path, "img")
        self.log_path_nml = pjoin(self.log_path, "nml")
        self.log_path_mat = pjoin(self.log_path, "mat")
        os.makedirs(self.log_path_img, exist_ok=True)
        os.makedirs(self.log_path_nml, exist_ok=True)
        os.makedirs(self.log_path_mat, exist_ok=True)

    def save_checkpoints(self, ckpt, epoch):
        savepath = os.path.join(self.log_path, 'model_params_%05d.pth' % epoch)
        torch.save(ckpt, savepath)
        print('Saved checkpoints at', savepath)
        if self.cfg.logger_type == "wandb":
            wandb.save(savepath)
    
    def log_status(self, epoch, metrics, status):
        if self.cfg.logger_type == "wandb":
            if status == "train":
                wandb.log({'epoch':epoch,
                        'train/rgb_loss': metrics["rgb_loss"]})
                wandb.log({'epoch':epoch,
                        'train/nml_mae': metrics["nml_mae"]})
                wandb.log({'epoch':epoch,
                        'train/lgt_mae': metrics["lgt_mae"]})
                wandb.log({'epoch':epoch,
                        'train/lgt_int': metrics["lgt_int"]})
            elif status == "eval":
                wandb.log({'epoch':epoch,
                        'eval/rgb_loss': metrics["rgb_loss"]})
                wandb.log({'epoch':epoch,
                        'eval/nml_mae': metrics["nml_mae"]})
                wandb.log({'epoch':epoch,
                        'eval/lgt_mae': metrics["lgt_mae"]})
                wandb.log({'epoch':epoch,
                        'eval/lgt_int': metrics["lgt_int"]})
                cv.imwrite(pjoin(self.log_path_img, "img.png"), metrics['plots']['img'])
                cv.imwrite(pjoin(self.log_path_nml, "nml.png"), metrics['plots']['nml'])
                cv.imwrite(pjoin(self.log_path_mat, "mat.png"), metrics['plots']['mat'])
        if status == "test":
            cv.imwrite(pjoin(self.log_path_nml, "est_nml.png"), metrics['plots']['est_nml'])
            cv.imwrite(pjoin(self.log_path_nml, "est_nml_err.png"), metrics['plots']['est_nml_err'])