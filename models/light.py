import torch
import torch.nn as nn
import torch.nn.functional as F


def activation(afunc='LReLU'):
    if afunc == 'LReLU':
        return nn.LeakyReLU(0.1, inplace=True)
    elif afunc == 'ReLU':
        return nn.ReLU(inplace=True)
    else:
        raise Exception('Unknown activation function')


def conv_layer(batchNorm, cin, cout, k=3, stride=1, pad=-1, afunc='LReLU'):
    if type(pad) != tuple:
        pad = pad if pad >= 0 else (k - 1) // 2
    mList = [nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True)]
    if batchNorm:
        print('=> convolutional layer with batchnorm')
        mList.append(nn.BatchNorm2d(cout))
    mList.append(activation(afunc))
    return nn.Sequential(*mList)


class ParaLight(nn.Module):
    def __init__(self, num_rays, light_init, requires_grad=True):
        super(ParaLight, self).__init__()
        light_direction_xy = light_init[0][:, :-1].clone().detach()
        light_direction_z = light_init[0][:, -1:].clone().detach()
        light_intensity = light_init[1].mean(dim=-1, keepdims=True).clone().detach()

        self.light_direction_xy = nn.Parameter(light_direction_xy.float(), requires_grad=requires_grad)
        self.light_direction_z = nn.Parameter(light_direction_z.float(), requires_grad=requires_grad)
        self.light_intensity = nn.Parameter(light_intensity.float(), requires_grad=requires_grad)

        self.num_rays = num_rays

    def forward(self, idx):
        out_ld = torch.cat([self.light_direction_xy[idx], -torch.abs(self.light_direction_z[idx])], dim=-1)
        out_ld = F.normalize(out_ld, p=2, dim=-1)[:, None, :]  # (96, 1, 3)

        out_ld = out_ld.repeat(1, self.num_rays, 1)
        out_ld = out_ld.view(-1, 3)  # (96*num_rays, 3)

        out_li = torch.abs(self.light_intensity[idx])[:, None, :]  # (96, 1, 1)
        out_li = out_li.repeat(1, self.num_rays, 3)
        out_li = out_li.view(-1, 3)  # (96*num_rays, 3)
        return out_ld, out_li

    def get_light_from_idx(self, idx):
        out_ld_r, out_li_r = self.forward(idx)
        return out_ld_r, out_li_r

    def get_all_lights(self):
        with torch.no_grad():
            light_direction_xy = self.light_direction_xy
            light_direction_z = -torch.abs(self.light_direction_z)
            light_intensity = torch.abs(self.light_intensity).repeat(1, 3)

            out_ld = torch.cat([light_direction_xy, light_direction_z], dim=-1)
            out_ld = F.normalize(out_ld, p=2, dim=-1)  # (96, 3)
            return out_ld, light_intensity


class ParaLightCNN(nn.Module):
    def __init__(
            self,
            num_layers=3,
            hidden_size=64,
            output_ch=4,
            batchNorm=False
    ):
        super(ParaLightCNN, self).__init__()
        self.conv1 = conv_layer(batchNorm, 4, 64,  k=3, stride=2, pad=1, afunc='LReLU')
        self.conv2 = conv_layer(batchNorm, 64, 128,  k=3, stride=2, pad=1)
        self.conv3 = conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.conv4 = conv_layer(batchNorm, 128, 128,  k=3, stride=2, pad=1)
        self.conv5 = conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.conv6 = conv_layer(batchNorm, 128, 256,  k=3, stride=2, pad=1)
        self.conv7 = conv_layer(batchNorm, 256, 256,  k=3, stride=1, pad=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = torch.nn.functional.relu
        self.dir_linears = nn.ModuleList(
            [nn.Linear(256, hidden_size)] + [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_linear = nn.Linear(hidden_size, output_ch)

    def forward(self, inputs):
        x = inputs
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        for i, l in enumerate(self.dir_linears):
            out = self.dir_linears[i](out)
            out = self.relu(out)
        outputs = self.output_linear(out)

        light_direction_xy = outputs[:, :2]
        light_direction_z = -torch.abs(outputs[:, 2:3])-0.1
        light_intensity = torch.abs(outputs[:, 3:])

        out_ld = torch.cat([light_direction_xy, light_direction_z], dim=-1)
        out_ld = F.normalize(out_ld, p=2, dim=-1)  # (96, 3)
        out_li = light_intensity  # (96, 1)

        outputs = {}
        outputs['dirs'] = out_ld
        outputs['ints'] = out_li
        return outputs

    def set_images(self, num_rays, images, device):
        self.num_rays = num_rays
        self.images = images
        self.device = device
        return

    def get_light_from_idx(self, idx):
        if hasattr(self, 'explicit_model'):
            out_ld_r, out_li_r = self.explicit_model(idx)
        else:
            x = self.images[idx].to(self.device)
            outputs = self.forward(x)
            out_ld, out_li = outputs['dirs'], outputs['ints'].repeat(1, 3)

            num_rays = self.num_rays
            out_ld_r = out_ld[:, None, :].repeat(1, num_rays, 1)  # (96, num_rays, 3)
            out_ld_r = out_ld_r.view(-1, 3)  # (96*num_rays, 3)

            out_li_r = out_li[:, None, :].repeat(1, num_rays, 1)
            out_li_r = out_li_r.view(-1, 3)  # (96*num_rays, 3)
        return out_ld_r, out_li_r

    def get_all_lights(self):
        if hasattr(self, 'explicit_model'):
            out_ld, out_li = self.explicit_model.get_all_lights()
        else:
            inputs = self.images.to(self.device)
            outputs = self.forward(inputs)
            out_ld, out_li = outputs['dirs'], outputs['ints']
        return out_ld, out_li

    def init_explicit_lights(self, explicit_direction=False, explicit_intensity=False):
        if explicit_direction or explicit_intensity:
            light_init = self.get_all_lights()
            self.explicit_intensity = explicit_intensity
            self.explicit_direction = explicit_direction
            self.explicit_model = ParaLight(self.num_rays, light_init, requires_grad=True)
        else:
            return
     
