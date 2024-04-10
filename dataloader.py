import torch
from torch.utils.data import Dataset
import numpy as np
import utils.dataset_loader.load_diligent as load_diligent
import cv2 as cv


class UPSDataset(Dataset):
    def __init__(self, data_dict, 
                       obj_name  ='ball', 
                       dataset   ='DiLiGenT', 
                       gray_scale=False, 
                       data_len  =1):
        # Images Handling.
        self.images = torch.tensor(data_dict['images'], dtype=torch.float32)  # (num_images, height, width, channel)
        if gray_scale:
            self.images = self.images.mean(dim=-1, keepdim=True)  # (num_images, height, width, 1)
            self.light_intensity = self.light_intensity.mean(dim=-1, keepdim=True)
        num_im, H, W = self.images.size(0), self.images.size(1), self.images.size(2)
        
        # Mask.
        self.mask = torch.tensor(data_dict['mask'], dtype=torch.float32)
        masks = self.mask[None,...].repeat((num_im, 1, 1))  # (num_images, height, width)
        self.o_mask = self._get_outer_contour(self.mask.numpy())
        self.o_mask = torch.from_numpy(self.o_mask)

        self.idx   = torch.where(self.mask > 0.5)
        self.idxs  = torch.where(masks > 0.5)
        self.o_idx = torch.where(self.o_mask > 0.5)
        
        # Normal Handling.
        self.gt_nml  = torch.tensor(data_dict['gt_normal'], dtype=torch.float32)
        self.cnt_nml, cnt = self._compute_contour_normal(data_dict['mask'])
        
        ## Particularly Handling for DiLiGenT100's non-occluding boundary.
        if dataset == "DiLiGenT100" and obj_name.split("_")[0] in ["BUNNY", "HEXAGON", "NUT", "PENTAGON", "PROPELLER", "SQUARE", "TURBINE"]:
            self.cnt_nml = torch.zeros_like(self.cnt_nml)
            self.cnt_nml[..., 2] = torch.tensor(-1.)
            self.cnt_nml = self.cnt_nml * cnt

        # For data.
        valid_cord = torch.stack([self.idx[1] / W, 
                                  self.idx[0] / H], dim=-1)
        self.valid_ocord = torch.stack([self.o_idx[1] / W, 
                                        self.o_idx[0] / H], dim=-1)
        
        # Problem of Cord
        valid_cord_max, _ = self.valid_ocord.max(dim=0)
        valid_cord_min, _ = self.valid_ocord.min(dim=0)

        self.bbox_uv    = [valid_cord_max,
                           valid_cord_min]
        self.bbox_int   = self._get_bounding_box_int()

        self.mean_cord  = self.valid_ocord.mean(0, keepdim=True)
        

        self.valid_ocord = self.valid_ocord - self.mean_cord
        self.gt_rgb = self.images[self.idxs].view(num_im, -1, 3)
        self.gt_ldir= torch.tensor(data_dict['light_direction'], dtype=torch.float32)
        self.gt_lint= torch.tensor(data_dict['light_intensity'], dtype=torch.float32)
        self.gt_nml = self.gt_nml[self.idx]
        
        self.data_len = min(data_len, num_im)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self._get_testing_rays(idx)

    def _get_testing_rays(self, ith):
        sample = {'gt_rgb':  self.gt_rgb[ith],
                  'gt_nml':  self.gt_nml,
                  'gt_ldir': self.gt_ldir,
                  'gt_lint': self.gt_lint,
                  'cnt_nml': self.cnt_nml,
                  'mean_uv': self.mean_cord,
                  'uv':      self.valid_ocord,
                  'idx':     ith}
        return sample
    
    def _get_bounding_box_int(self):
        mask = self.mask.numpy()
        valididx = np.where(mask > 0.5)
        xmin = valididx[0].min()
        xmax = valididx[0].max()
        ymin = valididx[1].min()
        ymax = valididx[1].max()

        xmin = max(0, xmin - 1)
        xmax = min(xmax + 2, mask.shape[0])
        ymin = max(0, ymin - 1)
        ymax = min(ymax + 2, mask.shape[1])
        return xmin, xmax, ymin, ymax
    
    def _get_outer_contour(self, mask):
        dilation = cv.dilate(mask, np.ones((3, 3)), iterations = 1)
        return dilation

    def _compute_contour_normal(self, _mask):
        blur = cv.GaussianBlur(_mask, (11, 11), 0)
        n_x = -cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=11, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        n_y = -cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=11, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

        n = np.sqrt(n_x**2 + n_y**2) + 1e-5
        contour_normal = np.zeros((_mask.shape[0], _mask.shape[1], 3), np.float32)
        contour_normal[:, :, 0] = n_x / n
        contour_normal[:, :, 1] = n_y / n
        contour_normal = torch.tensor(contour_normal, dtype=torch.float32)

        mask_x1, mask_x2, mask_y1, mask_y2 = self.mask.clone(), self.mask.clone(), self.mask.clone(), self.mask.clone()
        mask_x1[:-1, :] = self.mask[1:, :]
        mask_x2[1:, :]  = self.mask[:-1, :]
        mask_y1[:, :-1] = self.mask[:, 1:]
        mask_y2[:, 1:]  = self.mask[:, :-1]
        mask_1 = mask_x1 * mask_x2 * mask_y1 * mask_y2
        idxp_contour = torch.where((mask_1 < 0.5) & (self.mask > 0.5))

        contour_map = torch.zeros_like(self.mask)
        contour_map[idxp_contour] = 1
        contour = contour_map[self.idx]
        return contour[:, None] * contour_normal[self.idx], contour[:, None]

    def get_affix(self):
        x_max, x_min = max(self.idx[0]), min(self.idx[0])
        y_max, y_min = max(self.idx[1]), min(self.idx[1])

        x_max, x_min = min(x_max+15, self.images.shape[1]), max(x_min-15, 0)
        y_max, y_min = min(y_max+15, self.images.shape[2]), max(y_min-15, 0)

        out_images = self.images[:, x_min:x_max, y_min:y_max, :].permute([0,3,1,2])
        out_masks = self.mask[x_min:x_max, y_min:y_max][None, None, ...].repeat(out_images.size(0),1,1,1)
        out = torch.cat([out_images, out_masks], dim=1)
        return {"mask": self.mask,
                "o_mask": self.o_mask,
                "bbox_uv": self.bbox_uv,
                "bbox_int": self.bbox_int,
                "mask_img": out}