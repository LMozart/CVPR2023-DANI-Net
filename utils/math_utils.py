import math
import torch
import numpy as np


def cal_mae(gt_l, pred_l):
    dot_product = (gt_l * pred_l).sum(-1).clamp(-1, 1)
    angular_err = torch.acos(dot_product) * 180.0 / math.pi
    l_err_mean = angular_err.mean()
    return l_err_mean.item()


def cal_ints_acc(gt_i, pred_i):
    # Red channel:
    gt_i_c = gt_i[:, :1]
    pred_i_c = pred_i[:, :1]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio1 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)
    # Green channel:
    gt_i_c = gt_i[:, 1:2]
    pred_i_c = pred_i[:, 1:2]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio2 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)
    # Blue channel:
    gt_i_c = gt_i[:, 2:3]
    pred_i_c = pred_i[:, 2:3]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio3 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)

    ints_ratio = (ints_ratio1 + ints_ratio2 + ints_ratio3) / 3
    return ints_ratio.mean().item()


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def totalVariation(image, mask, num_rays):
    pixel_dif1 = torch.abs(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.abs(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var


def totalVariation_L2(image, mask, num_rays):
    pixel_dif1 = torch.square(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.square(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # the input tensor is added to the positional encoding if include_input=True
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )
