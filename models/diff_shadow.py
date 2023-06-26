from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F


# Code borrowed from: https://github.com/junxuan-li/Neural-Reflectance-PS
def _sample_points_along_light(img_xyz, light_xyz, num_points, bounding_box_xy):
    """
    Sample points from image coordinates img_xyz , toward light_xyz direction, until hit the image boundry
    image boundry is defined as   x: 0~1 ,  y: 0~1
    Args:
        img_xyz: (batch, 3)
        light_xyz: (batch, 3)
        num_points:  int
    Returns:  (batch, num_points, 3)
    """
    bound_k = torch.zeros_like(img_xyz[:,0])

    if bounding_box_xy is not None:
        xmax, ymax = bounding_box_xy[0][0], bounding_box_xy[0][1]
        xmin, ymin = bounding_box_xy[1][0], bounding_box_xy[1][1]
    else:
        xmax, ymax = 1, 1
        xmin, ymin = 0, 0

    kx0 = (xmin - img_xyz[:,0]) / light_xyz[:,0]
    yx0 = img_xyz[:,1] + kx0 * light_xyz[:,1]
    bound_k = torch.where((kx0 > 0) & (yx0 > ymin) & (yx0 < ymax), kx0, bound_k)

    kx1 = (xmax - img_xyz[:,0]) / light_xyz[:,0]
    yx1 = img_xyz[:,1] + kx1 * light_xyz[:,1]
    bound_k = torch.where((kx1 > 0) & (yx1 > ymin) & (yx1 < ymax), kx1, bound_k)

    ky0 = (ymin - img_xyz[:,1]) / light_xyz[:,1]
    xy0 = img_xyz[:,0] + ky0 * light_xyz[:,0]
    bound_k = torch.where((ky0 > 0) & (xy0 > xmin) & (xy0 < xmax), ky0, bound_k)

    ky1 = (ymax - img_xyz[:,1]) / light_xyz[:,1]
    xy1 = img_xyz[:,0] + ky1 * light_xyz[:,0]
    bound_k = torch.where((ky1 > 0) & (xy1 > xmin) & (xy1 < xmax), ky1, bound_k)
    k_steps = torch.logspace(start=0, end=1, steps=num_points+2, base=2, dtype=bound_k.dtype, device=bound_k.device)[1:-1] - 1

    temp = (bound_k[...,None] * k_steps[None,...])  # (batch, 1)*(1, num_points)
    boundry = img_xyz[:,None,:] + temp[..., None] * light_xyz[:,None,:]
    #          (batch, 1, 3)      (batch, num_points, 1) (batch, 1, 3)
    return boundry

def _mask_sample_points(mask, sample_points):
    """
    Mask out the sample_points, where its (x,y) coordinates is outside of the mask region
    Args:
        mask:  (H, W)   with 0 indicate out, and 1 indicate valid points
        sample_points: (batch, num_points, 3)
    Returns: (batch, num_points)  with 0 indicate out, and 1 indicate valid points
    """
    batch, num_points = sample_points.size(0), sample_points.size(1)
    sample_points_xy = sample_points.view(-1, 3)[:, :2].clone()
    sample_points_xy[:, 0] = (sample_points_xy[:, 0] - 0.5) * 2
    sample_points_xy[:, 1] = (sample_points_xy[:, 1] - 0.5) * 2   # convert to range -1~1
    sample_points_xy = sample_points_xy[None, None, ...]  # (1,1,batch*num_points,2)
    mask_value = F.grid_sample(mask[None, None, ...], sample_points_xy, mode='bilinear', padding_mode='border', align_corners=False)  # (1,1,1,batch*num_points)

    return mask_value.view(batch, num_points)

# Papers Implementation.
def _grid_interpolation(valid_points, depth_map, sigma, beta):
    """ Grid Interpolation.
    Args: 
        valid_points: (num_valid_pts, 3)
        depth_map:    depths map for interpolation
        sigma:        shadow coefficients 1
        beta:         shadow coefficients 2
    Returns: Shadow values.
    """
    light_z  = valid_points[:, -1:]
    sample_points_xy = valid_points.view(-1, 3)[:, :2].clone()
    sample_points_xy[:, 0] = (sample_points_xy[:, 0] - 0.5) * 2
    sample_points_xy[:, 1] = (sample_points_xy[:, 1] - 0.5) * 2   # convert to range -1~1
    sample_points_xy = sample_points_xy[None, None, ...]          # (1, 1, batch * num_points, 2)
    depth_interpolate = F.grid_sample(depth_map[None, None, ...], sample_points_xy, mode='bilinear', padding_mode='border', align_corners=False)  # (1, 1, 1, batch*num_points)
    out = torch.sigmoid((-light_z + depth_interpolate[0, 0, 0, ..., None]) / (sigma + 1e-5) + beta)
    return out
    
def differentiable_shadow(xyz, ld, depth, bounding_box, mask, params):
    """ Shadow Calculation. 
    Args: 
        xyz:   query points' coordinates
        ld:    light direction
        depth: depth map for interpolation
        params:  shadow coefficients (sigma & beta)
    Returns: Shadow values.
    """
    bd_xy        = bounding_box[0], bounding_box[1]
    sigma, beta  = params[0], params[1]
    sample_pts   = _sample_points_along_light(xyz, ld, 64, bd_xy)
    
    m = _mask_sample_points(mask, sample_pts)  # (batch, num_points)
    
    valid_pts    = sample_pts[torch.where(m > 0.9)]
    shadow_model = _grid_interpolation(valid_pts, depth, sigma, beta)
    sampled_pts_shadow_mask = torch.ones_like(m)
    sampled_pts_shadow_mask[torch.where(m > 0.9)] = shadow_model[..., 0] # (batch, num_points)
    out = torch.min(sampled_pts_shadow_mask, -1).values
    return out[..., None]


class ShadowParams(nn.Module):
    def __init__(self, sigma=0.6, beta=3., requires_grad=True):
        super(ShadowParams, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=requires_grad)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=requires_grad)

    def forward(self):
        return (1/torch.exp(10. * torch.abs(self.sigma).clip(1e-5, 1e5))), self.beta