from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SaveFeature():
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def remove(self):
        self.hook.remove()


class regression(nn.Module):

    def __init__(self):
        super(regression, self).__init__()

        # mobilenet = models.mobilenet_v2(pretrained=False)
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[0:-2]
        # self.conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*modules)
        self.features = SaveFeature(self.resnet[7][1].relu)

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, x):

        # x = self.conv(x)
        conv_features = self.resnet(x)
        x = F.adaptive_avg_pool2d(conv_features, 1)
        # x = F.avg_pool2d(conv_features, 16)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, self.features.features


# class MSELoss(nn.Module):
#
#     def __init__(self, use_target_weight):
#         super(MSELoss, self).__init__()
#
#
#     def forward(self, ):
#
#
#         return


class SaveFeature():

    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def remove(self):
        self.hook.remove()


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()
    print('model loading')


if __name__ == '__main__':

    img = torch.randn(3, 3, 512, 512)
    net = regression()
    out = net(img)
    print(out.size())
