from models.mlps import *
from models.asg  import *
from models.light import *
from models.diff_shadow import *
from utils.math_utils import *
from utils.record_utils import process_images
import torch.optim as optim

import os
import glob
import numpy as np
import cv2 as cv


class DepthBaseRunner(object):
    def __init__(self, cfg, device, affix) -> None:
        # Build MLPs & Models.
        self._build_model(cfg, device, affix)
        params_list = []
        params_list.append({'params': self.depth_mlp.parameters()})
        params_list.append({'params': self.mat_mlp.parameters()})
        params_list.append({'params': self.asg_model.parameters()})
        params_list.append({'params': self.light_model.parameters()})
        params_list.append({'params': self.params.parameters()})

        end_epoch = cfg.experiment.end_epoch

        self.optimizer = optim.Adam(params_list, lr=cfg.optimizer.lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lr_lambda=lambda step: cosine_annealing(step, end_epoch, 1, 0.1))
        
        # RGB Functions.
        if cfg.loss.rgb_loss == 'l1':
            self.rgb_loss_function = F.l1_loss
        elif cfg.loss.rgb_loss == 'l2':
            self.rgb_loss_function = F.mse_loss
        else:
            raise AttributeError('Undefined rgb loss function.')
        
        # setup
        self.cfg = cfg
        self.device = device

        # Grasp affix
        self.mask  = affix["mask"]
        self.o_mask= affix["o_mask"]
        self.bbox_uv = affix["bbox_uv"]
        self.bbox_int= affix["bbox_int"]

        self.H = self.mask.size(0)
        self.W = self.mask.size(1)
        self.num_bases = cfg.models.specular.num_bases
        self.num_o_rays= np.count_nonzero(affix["o_mask"])
        
        dx = 1 / self.mask.size(1)
        dy = 1 / self.mask.size(0)
        self.px = torch.zeros((1, 3), device=device)
        self.px[:, 0] = 1 * dx
        self.py = torch.zeros((1, 3), device=device)
        self.py[:, 1] = 1 * dy

        self.idxp  = torch.where(self.mask > 0.5)
        self.oidxp = torch.where(self.o_mask > 0.5)
        self.widxp = [i for i in range(len(self.oidxp[0])) 
                        if ((self.o_mask[self.oidxp[0][i], self.oidxp[1][i]] + 
                             self.mask[self.oidxp[0][i], self.oidxp[1][i]]) == 2)]
        self.idxp_invalid = np.where(self.mask.cpu().numpy() < 0.5)
        
        self.valid_cord = torch.stack([self.idxp[1] / self.W, 
                                       self.idxp[0] / self.H], dim=-1)
        
        self.mask = self.mask.to(device)
        self.o_mask = self.o_mask.to(device)
    
    def _build_model(self, cfg, device, affix):
        # Build MaterialMLP
        sph_bases_dim = cfg.models.specular.num_bases * 4
        
        mat_encode_fn = get_embedding_function(num_encoding_functions=cfg.models.materialmlp.num_encoding_fn_input)

        self.mat_mlp = MaterialMLP(num_layers=cfg.models.materialmlp.num_layers,
                                   hidden_size=cfg.models.materialmlp.hidden_size,
                                   skip_connect_every=cfg.models.materialmlp.skip_connect_every,
                                   encode_fn=mat_encode_fn,
                                   num_encoding_fn_input=cfg.models.materialmlp.num_encoding_fn_input,
                                   include_input_input=cfg.models.materialmlp.include_input_input,
                                   output_ch=sph_bases_dim)
        self.mat_mlp.train()
        self.mat_mlp.to(device)

        # Build DepthMLP
        depth_encode_fn = get_embedding_function(num_encoding_functions=cfg.models.depthmlp.num_encoding_fn_input)

        self.depth_mlp = DepthMLP(num_layers=cfg.models.depthmlp.num_layers,
                                  hidden_size=cfg.models.depthmlp.hidden_size,
                                  skip_connect_every=cfg.models.depthmlp.skip_connect_every,
                                  num_encoding_fn_input=cfg.models.depthmlp.num_encoding_fn_input,
                                  include_input_input=cfg.models.depthmlp.include_input_input,
                                  encode_fn=depth_encode_fn)
        self.depth_mlp.train()
        self.depth_mlp.to(device)

        # Build ASG
        if hasattr(cfg.models.specular, 'trainable_k'):
            trainable_k = cfg.models.specular.trainable_k
        else:
            trainable_k = True
        self.asg_model = ASG(num_bases=cfg.models.specular.num_bases,
                             k_low=cfg.models.specular.k_low,
                             k_high=cfg.models.specular.k_high,
                             trainable_k=trainable_k)
        self.asg_model.train()
        self.asg_model.to(device)

        # Build Light
        if cfg.models.light_model.type == 'Other Init':
            light_init_dirs = np.load(cfg.models.light_model.pre_dirs_path)
            light_init_ints = np.load(cfg.models.light_model.pre_ints_path)
            light_init_dirs[..., 1:] = -light_init_dirs[..., 1:]
            light_init = [torch.from_numpy(light_init_dirs), torch.from_numpy(light_init_ints)]
            
            self.light_model = ParaLight(light_init=light_init, 
                                         requires_grad=True)
            self.light_model.to(device)
        elif cfg.models.light_model.type == 'CNN Init':
            self.light_model = ParaLightCNN(
                num_layers=cfg.models.light_model.num_layers,
                hidden_size=cfg.models.light_model.hidden_size,
                output_ch=4,
                batchNorm=False
            )
            self.light_model.train()
            self.light_model.to(device)
        else:
            raise NotImplementedError('Unknown Light Model')
        
        if cfg.models.light_model.load_pretrain:
            model_checkpoint_pth = os.path.expanduser(cfg.models.light_model.load_pretrain)
            if model_checkpoint_pth[-4:] != '.pth':
                model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
            print('[INFO] Found pretrain light model checkpoints: ', model_checkpoint_pth)
            ckpt = torch.load(model_checkpoint_pth, map_location=device)
            self.light_model.load_state_dict(ckpt['model_state_dict'])
            self.light_model.set_images(
                num_rays=np.count_nonzero(affix['mask']),
                images=affix["mask_img"],
                device=device,
            )
            self.light_model.init_explicit_lights(
                explicit_direction=cfg.models.light_model.explicit_direction,
                explicit_intensity=cfg.models.light_model.explicit_intensity,
            )

        # Shadows
        self.params = ShadowParams(sigma=cfg.models.shadow.sigma,
                                   beta=cfg.models.shadow.beta,
                                   requires_grad=cfg.models.shadow.trainable_sigma)

    def load_checkpoints(self, cfg):
        model_checkpoint_pth = os.path.expanduser(cfg.models.checkpoint_path)
        if model_checkpoint_pth[-4:] != '.pth':
            model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
        print(f'Found checkpoints: {model_checkpoint_pth}.')
        ckpt = torch.load(model_checkpoint_pth, map_location=self.device)

        self.depth_mlp.load_state_dict(ckpt['depth_state_dict'])
        self.mat_mlp.load_state_dict(ckpt['material_state_dict'])
        self.asg_model.load_state_dict(ckpt['specular_model_state_dict'])
        self.light_model.load_state_dict(ckpt['light_model_state_dict'])
        
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
    def fetch_checkpoints(self, epoch):
        return {'global_step': epoch,
                'depth_state_dict': self.depth_mlp.state_dict(),
                'material_state_dict': self.mat_mlp.state_dict(),
                'specular_model_state_dict': self.asg_model.state_dict(),
                'sigma_state_dict': self.params.state_dict(),
                'light_model_state_dict': self.light_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()}

    def set_model_status(self, status):
        if status == "train":
            self.depth_mlp.train()
            self.mat_mlp.train()
        elif status == "eval":
            self.depth_mlp.eval()
            self.mat_mlp.eval()

    def model_pred(self, in_batch, epoch, end_epoch):
        uv = in_batch["uv"][0].to(self.device)
        bs = len(in_batch["idx"])
        idx= in_batch["idx"].to(self.device)
        mean_uv = in_batch["mean_uv"][0].to(self.device)
        
        est_depth = self.depth_mlp(uv)
        est_coeff = self.mat_mlp(uv)
        est_s_param = self.params()

        est_ldir, est_lint = self.light_model.get_light_from_idx(idx=idx)
        est_ldir_all, est_lint_all = self.light_model.get_all_lights()

        # Shape & Shading Calculation.
        depth_map = torch.zeros((self.H, self.W), 
                                dtype=torch.float32, 
                                device=self.device)
        
        depth_map[self.oidxp] = est_depth.view(-1)
        est_nml_0  = self._nml_fit(depth_map)[self.widxp, :]
        est_nml    = est_nml_0.repeat(bs, 1)
        est_shad   = F.relu((est_nml * est_ldir).sum(dim=-1, keepdims=True))

        # Shadow Calculation
        z   = depth_map[self.idxp][..., None]
        xyz = torch.cat((uv[self.widxp], z), -1)
        xyz[:, :2] = xyz[:, :2] + mean_uv.to(self.device)
        xyz = xyz.repeat(bs, 1)

        est_s= differentiable_shadow(xyz=xyz,
                                     ld=est_ldir,
                                     depth=depth_map,
                                     bounding_box=self.bbox_uv,
                                     mask=self.mask,
                                     params=est_s_param)
        
        # BRDF Calculation.
        est_diff_0 = est_coeff['diff'][self.widxp, :]
        est_spec_coeff_0 = est_coeff['spec_coeff'][self.widxp, :]

        est_diff = est_diff_0.repeat(bs, 1)
        est_spec_coeff = est_spec_coeff_0.repeat(bs, 1)
        
        est_spec_coeff = est_spec_coeff[..., self.num_bases:].view(-1, self.num_bases, 3)
        est_spec_coeff = dynamic_basis(est_spec_coeff, epoch, end_epoch, self.num_bases)
        est_spec = self.asg_model(light=est_ldir, 
                                  normal=est_nml,
                                  mu=est_spec_coeff)
        est_spec = est_spec.sum(dim=1)
        est_brdf = est_diff + est_spec
        
        # Render Equation.
        est_rgb  = est_brdf * est_shad * est_lint * est_s

        pred_dict = {"est_s": est_s,
                     "est_shad": est_shad,
                     "est_ldir": est_ldir_all,
                     "est_lint": est_lint_all,
                     "est_rgb": est_rgb,
                     "est_nml": est_nml_0,
                     "est_diff": est_diff_0,
                     "est_s_param": est_s_param,
                     "est_spec_coeff": est_spec_coeff_0,
                     "est_spec": est_spec * est_shad,
                     "est_brdf": est_brdf,
                     "depth_map": depth_map}
        return pred_dict

    def _nml_fit(self, depth_map):
        dl = torch.zeros_like(depth_map, device=self.device)
        dr = torch.zeros_like(depth_map, device=self.device)
        dt = torch.zeros_like(depth_map, device=self.device)
        db = torch.zeros_like(depth_map, device=self.device)

        dr[:, 1:-1] = (depth_map[:, 2:] - depth_map[:, 1:-1]) * self.o_mask[:, 2:] * self.o_mask[:, 1:-1]
        dl[:, 1:-1] = (depth_map[:, 1:-1] - depth_map[:, :-2]) * self.o_mask[:, :-2] * self.o_mask[:, 1:-1]
        dt[1:-1, :] = (depth_map[2:, :] - depth_map[1:-1, :]) * self.o_mask[2:, :] * self.o_mask[1:-1, :]
        db[1:-1:, ] = (depth_map[1:-1, :] - depth_map[:-2, :]) * self.o_mask[:-2, :] * self.o_mask[1:-1, :]

        pr = self.px.to(self.device).clone().repeat(self.num_o_rays, 1)
        pl = self.px.to(self.device).clone().repeat(self.num_o_rays, 1)
        pt = self.py.to(self.device).clone().repeat(self.num_o_rays, 1)
        pb = self.py.to(self.device).clone().repeat(self.num_o_rays, 1)

        pr[:, 2] = dr[self.oidxp]
        pl[:, 2] = dl[self.oidxp]
        pt[:, 2] = dt[self.oidxp]
        pb[:, 2] = db[self.oidxp]

        nml_sq1 = F.normalize(-torch.cross(pr, pt, dim=-1), p=2, dim=-1)
        nml_sq2 = F.normalize(-torch.cross(pl, pt, dim=-1), p=2, dim=-1)
        nml_sq3 = F.normalize(-torch.cross(pl, pb, dim=-1), p=2, dim=-1)
        nml_sq4 = F.normalize(-torch.cross(pr, pb, dim=-1), p=2, dim=-1)

        sq1_dist = 1 / ((torch.abs(dr) + torch.abs(dt)) + 1e-5)
        sq2_dist = 1 / ((torch.abs(dl) + torch.abs(dt)) + 1e-5)
        sq3_dist = 1 / ((torch.abs(dl) + torch.abs(db)) + 1e-5)
        sq4_dist = 1 / ((torch.abs(dr) + torch.abs(db)) + 1e-5)

        sq_sum   = sq1_dist + sq2_dist + sq3_dist + sq4_dist

        sq1_ratio = (sq1_dist / sq_sum).detach()
        sq2_ratio = (sq2_dist / sq_sum).detach()
        sq3_ratio = (sq3_dist / sq_sum).detach()
        sq4_ratio = (sq4_dist / sq_sum).detach()

        est_nml = sq1_ratio[self.oidxp][..., None] * nml_sq1 + \
                  sq2_ratio[self.oidxp][..., None] * nml_sq2 + \
                  sq3_ratio[self.oidxp][..., None] * nml_sq3 + \
                  sq4_ratio[self.oidxp][..., None] * nml_sq4
        
        est_nml = F.normalize(est_nml, p=2, dim=-1)
        return est_nml

    def train(self, cfg, pred, data, end_epoch, epoch):
        num_rays = len(self.idxp[0])
        bs = len(data["idx"])
        
        data['gt_rgb'] = data['gt_rgb'].view(-1, 3).to(self.device)
        data['cnt_nml']= data['cnt_nml'].to(self.device)

        rgb_loss = self.rgb_loss_function(pred['est_rgb'], data['gt_rgb'].view(-1, 3))
        rgb_loss_val = rgb_loss.item()
        loss = rgb_loss

        if cfg.strategy.reg_anneal:
            anneal = cosine_annealing(epoch, cfg.loss.regularize_epoches * end_epoch, 1, 0)
        else:
            anneal = 1.

        if epoch <= int(cfg.loss.regularize_epoches * end_epoch):  # if epoch is small, use tv to guide the network
            if cfg.loss.diff_tv_factor > 0:
                diff_color_map = torch.zeros((self.H, self.W, 3),  dtype=torch.float32, device=self.device)
                diff_color_map[self.idxp] = pred['est_diff']
                tv_loss = totalVariation(diff_color_map, self.mask, num_rays) * bs * cfg.loss.diff_tv_factor * anneal
                loss += tv_loss
            if cfg.loss.spec_tv_factor > 0:
                spec_color_map = torch.zeros((self.H, self.W, pred['est_spec_coeff'].size(1)), dtype=torch.float32, device=self.device)
                spec_color_map[self.idxp] = pred['est_spec_coeff']
                tv_loss = totalVariation(spec_color_map, self.mask, num_rays) * bs * cfg.loss.spec_tv_factor * anneal
                loss += tv_loss
            if cfg.loss.normal_tv_factor > 0:
                normal_map = torch.zeros((self.H, self.W, 3), dtype=torch.float32, device=self.device)
                normal_map[self.idxp] = pred['est_nml']
                nml_loss = totalVariation_L2(normal_map, self.mask, num_rays) * bs * cfg.loss.normal_tv_factor * anneal
                loss += nml_loss
            if cfg.loss.depth_tv_factor > 0:
                dpth_loss = totalVariation(pred['depth_map'][..., None], self.mask, num_rays) * bs * cfg.loss.depth_tv_factor * anneal
                loss += dpth_loss
            if cfg.loss.spec_coeff_factor > 0:
                spec_coeff_loss = F.l1_loss(pred['est_spec_coeff'], torch.zeros_like(pred['est_spec_coeff']))
                loss += spec_coeff_loss * cfg.loss.spec_coeff_factor * bs * anneal
                
        if cfg.loss.contour_factor > 0 and epoch <= int(0.75 * end_epoch):
            normal_map = torch.zeros((self.H, self.W, 3), dtype=torch.float32, device=self.device)
            normal_map[self.idxp] = pred['est_nml']
            tv_loss = totalVariation_L2(normal_map, self.mask, num_rays) * bs * 0.01 * cfg.loss.contour_factor
            loss += tv_loss
        if cfg.dataset.occluding_bound:
            if cfg.loss.contour_factor > 0 and epoch <= int(0.75 * end_epoch):
                contour_normal_loss = 1 - torch.sum(pred['est_nml'] * data['cnt_nml'], dim=-1).mean()
                loss += contour_normal_loss * cfg.loss.contour_factor
        else:
            contour_normal_loss = 1 - torch.sum(pred['est_nml'] * data['cnt_nml'], dim=-1).mean()
            loss += contour_normal_loss * cfg.loss.contour_factor
        
        if cfg.loss.normalize_sigma_factor > 0:
            if cfg.strategy.sigma_l1_reg:
                loss += cfg.loss.normalize_sigma_factor * pred['est_s_param'][0]
            else:
                loss += cfg.loss.normalize_sigma_factor * pred['est_s_param'][0] ** 2
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metric = {'rgb_loss': rgb_loss_val,
                  'nml_mae': cal_mae(pred['est_nml'].detach().cpu(), data['gt_nml'][0]),
                  'lgt_mae': cal_mae(pred['est_ldir'].detach().cpu(), data['gt_ldir'][0]),
                  'lgt_int': cal_ints_acc(pred['est_lint'].detach().cpu(), data['gt_lint'][0])}
        return metric
    
    def eval(self, pred, data):
        rgb_loss = F.l1_loss(pred['est_rgb'].view(-1).detach().cpu(), data['gt_rgb'].view(-1))
        rgb_loss = rgb_loss.item()

        nml_mae  = cal_mae(pred['est_nml'].detach().cpu(), data['gt_nml'])
        
        plots = {}
        ######## Normal Estimation Plots ########
        temp_est_nml = pred['est_nml'].detach().clone()
        temp_gt_nml  = data['gt_nml'].detach().clone()

        temp_est_nml[..., 1:] = -temp_est_nml[..., 1:]
        temp_gt_nml[..., 1:]  = -temp_gt_nml[..., 1:]
        
        est_nml_map = process_images(pixels=(temp_est_nml + 1) / 2,
                       h=self.H, w=self.W, channel=3,
                       idxp=self.idxp,
                       idxp_invalid=self.idxp_invalid,
                       bounding_box_int=self.bbox_int)
        gt_nml_map  = process_images(pixels=(temp_gt_nml + 1) / 2,
                       h=self.H, w=self.W, channel=3,
                       idxp=self.idxp,
                       idxp_invalid=self.idxp_invalid,
                       bounding_box_int=self.bbox_int)
        
        normal_err = torch.arccos(torch.clamp((pred['est_nml'].detach().cpu() * data['gt_nml']).sum(dim=-1), max=1, min=-1)) / math.pi * 180
        nml_err_map = torch.zeros((self.H, self.W), dtype=torch.float32)
        nml_err_map[self.idxp] = torch.clamp(normal_err, max=90)
        nml_err_map = nml_err_map.numpy()
        nml_err_map = (np.clip(nml_err_map / 90, 0, 1) * 255).astype(np.uint8)
        nml_err_map = cv.applyColorMap(nml_err_map, colormap=cv.COLORMAP_JET)
        nml_err_map[self.idxp_invalid] = 255
        nml_err_map = nml_err_map[self.bbox_int[0]:self.bbox_int[1], self.bbox_int[2]-15:self.bbox_int[3]+15]
        
        nml_stack = np.concatenate((gt_nml_map, est_nml_map, nml_err_map), axis=1)
        plots["nml"] = nml_stack
        
        ######## Render Plots ########
        est_img = process_images(pixels=pred['est_rgb'][-len(self.idxp[0]):],
                                 h=self.H, w=self.W, channel=3,
                                 idxp=self.idxp,
                                 idxp_invalid=self.idxp_invalid,
                                 bounding_box_int=self.bbox_int)
        
        gt_img  = process_images(pixels=data['gt_rgb'][-len(self.idxp[0]):],
                                 h=self.H, w=self.W, channel=3,
                                 idxp=self.idxp,
                                 idxp_invalid=self.idxp_invalid,
                                 bounding_box_int=self.bbox_int)

        img_stack = np.concatenate((est_img, gt_img), axis=1)
        plots["img"] = img_stack

        diff_map  = process_images(pixels=pred['est_diff'] / pred['est_diff'].max(),
                                   h=self.H, w=self.W, channel=3,
                                   idxp=self.idxp,
                                   idxp_invalid=self.idxp_invalid,
                                   bounding_box_int=self.bbox_int)
        shadow_map = process_images(pixels=pred['est_s'][:len(self.idxp[0])],
                                    h=self.H, w=self.W, channel=3,
                                    idxp=self.idxp,
                                    idxp_invalid=self.idxp_invalid,
                                    bounding_box_int=self.bbox_int)
        spec_map   = process_images(pixels=pred['est_spec'][:len(self.idxp[0])],
                                    h=self.H, w=self.W, channel=3,
                                    idxp=self.idxp,
                                    idxp_invalid=self.idxp_invalid,
                                    bounding_box_int=self.bbox_int)
        shad_map   = process_images(pixels=pred['est_shad'][:len(self.idxp[0])],
                                    h=self.H, w=self.W, channel=1,
                                    idxp=self.idxp,
                                    idxp_invalid=self.idxp_invalid,
                                    bounding_box_int=self.bbox_int)
        shad_map   = process_images(pixels=pred['est_brdf'][:len(self.idxp[0])],
                                    h=self.H, w=self.W, channel=3,
                                    idxp=self.idxp,
                                    idxp_invalid=self.idxp_invalid,
                                    bounding_box_int=self.bbox_int)
        
        brdf_stack = np.concatenate((diff_map, shadow_map, spec_map, shad_map), axis=1)
        plots["mat"] = brdf_stack
        metric = {'rgb_loss': rgb_loss,
                  'nml_mae': nml_mae,
                  'lgt_mae': cal_mae(pred['est_ldir'].detach().cpu(), data['gt_ldir']),
                  'lgt_int': cal_ints_acc(pred['est_lint'].detach().cpu(), data['gt_lint'][0]),
                  'plots': plots}
        return metric