import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import os.path as osp

import argparse
import os
import time
import numpy as np
from dataset import IdDataset, DataLoader
from model_original import save_checkpoint, load_checkpoint, regression
from mobilenet_v3 import MobileNetV3_Small

from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import matplotlib.pyplot as plt
import cv2
import math
# import ipdb


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="test")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)

    parser.add_argument("--dataroot", default=r"F:")  # G:\KeypointsRegression\dataset
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default=r'G:\KeypointsRegression\test_result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default=r'G:\KeypointsRegression\checkpoints\Keypoints_regression\reg_final.pth',
                        help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def resize_and_pad(opt, img):

    # full_path = osp.join(r'G:\KeypointsRegression\val', path)
    # img = cv2.imread(full_path)
    h, w, c = img.shape
    if h > w:
        # resize the image to a fixed size
        ratio = opt.height / h
        img = cv2.resize(img, dsize=(int(ratio * w), int(ratio * h)))
        pad_bottom = opt.height - int(ratio * h)
        pad_left = (opt.width - int(ratio * w)) // 2
        pad_right = (opt.width - int(ratio * w)) - (opt.width - int(ratio * w)) // 2
        pad_up = 0
        img = np.pad(img, ((0, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0)

    else:
        ratio = opt.width / w
        img = cv2.resize(img, dsize=(int(ratio*w), int(ratio*h)))
        pad_right = opt.width - int(ratio*w)
        pad_up = (opt.height - int(ratio*h)) // 2
        pad_bottom = (opt.height - int(ratio*h)) - (opt.height - int(ratio*h)) // 2
        img = np.pad(img, ((pad_up, pad_bottom), (0, pad_right), (0, 0)), 'constant', constant_values=0)
        pad_left = 0

    return img


def evaluate(opt, test_loader, model, board):

    model.cuda()
    model.eval()

    total_time = 0
    # n_time_inference = 60
    total_error = 0
    for step, inputs in enumerate(test_loader.data_loader):

        # original_image = inputs['original_image']
        image = inputs['image'].cuda()
        im_name = inputs['im_name']
        gt_rad = inputs['rad']
        pad_up = inputs['pad_up'].cuda()
        pad_bottom = inputs['pad_bottom'].cuda()
        pad_left = inputs['pad_left'].cuda()
        pad_right = inputs['pad_right'].cuda()
        ratio = inputs['ratio'].cuda()

        iter_start_time = time.time()
        # keypoints = model(image.float())
        pred_rad, conv_features = model(image.float())
        # feature = activation[step]
        t = time.time() - iter_start_time

        total_time += t
        print('processing:', im_name[0])
        rad = pred_rad.clone().detach().cpu().numpy()
        gt_rad = gt_rad.clone().detach().cpu().numpy()
        # mapping the transformed coordinate to the original size
        # keypoints = mapping_coordinate(opt, keypoints, ratio, pad_up, pad_bottom, pad_left, pad_right)
        # save_image(opt, keypoints, pad_up, im_name)
        error = save_rotate_image(opt, rad, gt_rad, image, im_name)
        # generate_heatmap(opt, image, conv_features, im_name)
        total_error += error

    # avg_time = total_time / n_time_inference
    print("total error of the degree: ", total_error)


def mapping_coordinate(opt, keypoints, ratio, pad_up, pad_bottom, pad_left, pad_right):

    kpts = keypoints.clone().detach().cpu().numpy().tolist()
    kpts = kpts[0]
    radius = kpts.pop(-1)

    slope = math.tan(radius)
    neck_x = kpts[4]
    neck_y = kpts[5]
    bias = neck_y - slope*neck_x
    x = ((opt.height-pad_up)/opt.height - bias) / slope
    y = slope * x + bias
    kpts.insert(6, x)
    kpts.insert(7, y)

    # for i in range(len(kpts)):
    #     if i % 2 == 0:
    #         kpts[i] = int(kpts[i]*opt.height / ratio) - pad_left
    #     else:
    #         kpts[i] = int(kpts[i]*opt.width / ratio) - pad_up

    for i in range(len(kpts)):
        if i % 2 == 0:
            kpts[i] = int(kpts[i]*opt.height)
        else:
            kpts[i] = int(kpts[i]*opt.width)

    return kpts


def save_image(opt, keypoints, pad_up, im_name):

    kpts = keypoints.clone().detach().cpu().numpy().tolist()
    kpts = kpts[0]
    pd = pad_up.clone().detach().cpu().numpy()
    radius = kpts.pop(-1)

    slope = math.tan(radius)
    neck_x = kpts[4]
    neck_y = kpts[5]
    bias = neck_y - slope * neck_x
    x_coord = ((opt.height - pd)/opt.height - bias) / slope
    y_coord = x_coord * slope + bias
    kpts.insert(6, x_coord)
    kpts.insert(7, y_coord)

    x = []
    y = []
    for idx in range(len(kpts)):
        if idx % 2 == 0:
            x.append(kpts[idx])
        else:
            y.append(kpts[idx])

    assert len(x) == 10, 'outputs not equal to the 10 points'

    resize_image = resize_and_pad(opt, im_name[0])
    img = resize_image.copy()

    for i in range(len(x)):
        plot_image = cv2.circle(img, (int(x[i]*opt.height), int(y[i]*opt.width)), radius=0, color=(0, 0, 255), thickness=5)

    save_name = os.path.join(opt.result_dir, opt.name, im_name[0])
    cv2.imwrite(save_name, plot_image)


def save_rotate_image(opt, rad, gt_rad, image, im_name):

    img = image.clone().cpu().numpy()
    img = (img + 1)*0.5
    reshape_image = np.transpose(img[0, :, :, :], axes=[1, 2, 0])

    prefix, extension = im_name[0].split('_torso')
    name = prefix + extension
    # path = r'G:\KeypointsRegression\dataset\test'
    path = r'F:\train'
    original_image = cv2.imread(osp.join(path, name))
    if original_image is None:
        raise ValueError('failed to load image {}'.format(osp.join(path, name)))
    resize_image = resize_and_pad(opt, original_image)

    # rad dtype:
    pred_degree = np.rad2deg(rad)
    gt_degree = np.rad2deg(gt_rad)
    error = np.linalg.norm((gt_degree-pred_degree), ord=2)

    # print('degree is:', pred_degree)
    # M = cv2.getRotationMatrix2D((opt.width / 2, opt.height / 2), angle=float(pred_degree), scale=1)
    # dst = cv2.warpAffine(resize_image, M, (opt.width, opt.height))

    # rotate_name = im_name[0].split('.')[0] + '-rotate.jpg'
    # cv2.imwrite(osp.join(opt.result_dir, im_name[0]), original_image)
    # cv2.imwrite(osp.join(opt.result_dir, rotate_name), dst)

    return error


def generate_heatmap(opt, image, features, im_name):

    img = image.clone().cpu().numpy()
    img = (img + 1) * 0.5
    reshape_image = np.transpose(img[0, :, :, :], axes=[1, 2, 0])
    original_image = reshape_image[:, :, ::-1]*255

    features = torch.sum(features, dim=[0, 1])
    feat = features.clone().detach().cpu().numpy()
    # assert len(feat.size()) == 2, 'the shape of the feat is not two dimension'
    feat_norm = (feat - feat.min()) / (feat.max() - feat.min())
    feat_norm = cv2.resize(feat_norm, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    feat_array = np.array(feat_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(feat_array, cv2.COLORMAP_JET)
    heatmap = heatmap * 0.3 + 0.6 * original_image
    heatmap_name = im_name[0].split('.')[0] + '-heatmap.jpg'
    cv2.imwrite(osp.join(opt.result_dir, heatmap_name), heatmap)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def main():

    opt = get_opt()
    print(opt)

    # create dataset
    test_split = 'train'
    test_dataset = IdDataset(opt, test_split)

    # create dataloader
    test_loader = DataLoader(opt, test_dataset)

    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))  # writer = SummaryWriter
    model = regression()
    # model.resnet[7][1].relu.register_forward_hook(get_activation('relu'))

    load_checkpoint(model, opt.checkpoint)
    with torch.no_grad():
        evaluate(opt, test_loader, model, board)

    print('Finished!')


if __name__ == "__main__":
    main()