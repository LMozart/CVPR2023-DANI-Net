# CVPR2023-DANI-Net


**[DANI-Net: Uncalibrated Photometric Stereo by Differentiable Shadow Handling, Anisotropic Reflectance Modeling, and Neural Inverse Rendering](https://lmozart.github.io/CVPR2023-DANI-Net/)**
<br>
[Zongrui Li](https://github.com/LMozart), [Qian Zheng](https://person.zju.edu.cn/zq), [Boxin Shi](http://ci.idm.pku.edu.cn/), [Gang Pan](https://person.zju.edu.cn/en/gpan), [Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/)
<br>

Given a set of observed images captured under varying, parallel lights, DANI-Net recovers light conditions (directions and intensities), surface normal, anisotropic reflectance, and soft shadow map.
## Our Relighting Results
<p align="center">
    <img src='assets/relighting.gif'>
</p>

## Dependencies
We use Anaconda to install the dependencies given following code:
```shell
# Create a new python3.8 environment named dani-net
conda env create -f environment.yml
conda activate dani-net
```

## Train
### Train on benchmark datasets.
The datasets can be downloaded according to the table below:

|  Dataset   | Link  |
|  ----  | ----  |
| DiLiGenT Benchmark | [Link](https://sites.google.com/site/photometricstereodata/single) |
| DiLiGenT10^2 Benchmark | [Link](https://photometricstereo.github.io/diligent102.html) |
| Gourd & Apple| [Link](https://drive.google.com/drive/folders/1_bOM2nghnYTBrlmNOqRh5y5elPjvShOb?usp=sharing)|
| Light Stage Data Gallery | [Link](http://vgl.ict.usc.edu/Data/LightStage/) |

Please download and unzip it to the 'data' folder in the root directory. Please create the 'data' folder manually when you first use this code.To test DANI-Net on a particular object, you may run:
```shell
python train.py --config configs/diligent/YOUR_OBJ_NAME.yml --exp_code YOUR_EXP_TAG
```

To test DANI-Net on multiple objects in an particular dataset, please run:
```shell
# DiLiGenT
sh scripts/train_diligent.sh

# DiLiGenT 10^2
sh scripts/train_diligent100.sh

# Gourd & Apple
sh scripts/train_apple.sh

# Light Stage
sh train_lightstage.sh
```

## Test

We provide all the trained models in this [link](https://drive.google.com/drive/folders/1Z32BrBHluyETLE_VBmPcolSKm66dnGrP?usp=sharing), download and unzip them to the 'runs' folder in the root directory. Please create the 'runs' folder manually when you first use this code.
To test the results in an particular dataset, please run:
```shell
# DiLiGenT
sh scripts/test_diligent.sh

# DiLiGenT 10^2
sh scripts/test_diligent100.sh

# Gourd & Apple
sh scripts/test_apple.sh

# Light Stage
sh scripts/test_lightstage.sh
```

## Acknowledgement
Part of the code is based on [Neural-Reflectance-PS](https://github.com/junxuan-li/Neural-Reflectance-PS), [nerf-pytorch](https://github.com/krrish94/nerf-pytorch), [SCPS-NIR](https://github.com/junxuan-li/SCPS-NIR/) repository.

## Citation
    @inproceedings{li2023dani,
      title={DANI-Net: Uncalibrated Photometric Stereo by Differentiable Shadow Handling, Anisotropic Reflectance Modeling, and Neural Inverse Rendering},
      author={Li, Zongrui and Zheng, Qian and Shi, Boxin and Pan, Gang and Jiang, Xudong},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2023}}