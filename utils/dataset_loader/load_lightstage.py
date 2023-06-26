import cv2 as cv
import os
import numpy as np
import glob
import scipy.io as sio
from skimage.transform import resize
import OpenEXR, Imath

"""
we convert the normal and light direction for:
     --------->   x
    |   
    |  
    | 
    y    
x direction is looking right
y direction is looking down
z direction is looking into the image
"""

def get_channel_from_exr(exr, channel_name):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    arr_str = exr.channel(channel_name, pt)
    arr = np.frombuffer(arr_str, dtype=np.float32)
    arr.shape = (size[1], size[0])

    return arr

def data_from_exr(exr_file_path):
    f = OpenEXR.InputFile(exr_file_path)

    B = get_channel_from_exr(f, 'B')
    G = get_channel_from_exr(f, 'G')
    R = get_channel_from_exr(f, 'R')

    rgb = np.stack([R, G, B], axis=-1)
    return rgb

def parse_txt(filename):
    out_list = []
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        for x in lines:
            x = x.strip().split()[1:]
            lxyz = np.array([float(v) for v in x], dtype=np.float32)
            out_list.append(lxyz)
    out_arr = np.stack(out_list, axis=0).astype(np.float32)
    return out_arr


def load_lightstage(path, scale=1):
    images = []

    if 'helmet_front_left' in path:
        read_exr_file = True
    else:
        read_exr_file = False
    if ('knight_standing' in path) or ('knight_kneeling' in path):  # or 'knight_fighting' in path:
        scale = 1
    else:
        scale = 2
        
    if read_exr_file:
        img_file_list = sorted(glob.glob(os.path.join(path, "*[0-9]*.exr")))
    else:
        img_file_list = sorted(glob.glob(os.path.join(path,"*[0-9]*.png")))

    for img_file in img_file_list:
        if read_exr_file:
            img = data_from_exr(img_file)
        else:
            img = cv.imread(img_file)[:,:,::-1].astype(np.float32) / 255.
            img = img ** 2.2
        if scale!=1:
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)
        images.append(img)
    images = np.stack(images, axis=0)

    mask_files = glob.glob(os.path.join(path, "matte.png"))[0]
    mask_inv = cv.imread(mask_files, 0).astype(np.float32) / 255.
    mask = np.zeros_like(mask_inv)
    mask[np.where(mask_inv < 0.5)] = 1

    if scale!=1:
        mask = cv.resize(mask, dim, interpolation=cv.INTER_NEAREST)

    light_dir_files = os.path.join(os.path.dirname(path), "light_directions.txt")
    light_dir = parse_txt(light_dir_files)
    
    light_dir[..., 0] = -light_dir[..., 0]  # convert x-> -x
    light_intensity_files = os.path.join(os.path.dirname(path), "light_intensities.txt")
    light_intensity = parse_txt(light_intensity_files)

    gt_normal = np.zeros_like(images[0])

    idx = np.where(light_dir[..., -1] < -0.1)
    light_dir = light_dir[idx]
    light_intensity = light_intensity[idx]
    images = images[idx]
    print('Only use front size lights ld_z<-0.1, total images: %d' % len(light_dir))

    out_dict = {'images': images, 'mask': mask, 'light_direction': light_dir, 'light_intensity': light_intensity, 'gt_normal': gt_normal}
    return out_dict
