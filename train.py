import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

import argparse
from config import update_config, config
import os
import os.path as osp
import time
import numpy as np
from dataset import IdDataset, DataLoader
from model_original import save_checkpoint, load_checkpoint, regression
from mobilenet_v3 import MobileNetV3_Large
from mobilenet_v3 import MobileNetV3_Small
from hrnet_cls import get_cls_net
from visualization import board_add_images

from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import matplotlib.pyplot as plt
import cv2
import math
from loss import AverageMeter
# import ipdb
# from apex import amp


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Keypoints_regression")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=2)

    parser.add_argument("--dataroot", default=r"G:\KeypointsRegression\dataset")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--keep_step", type=int, default=11400)  # 总共50多个epoch 前一半个学习率保持不变
    parser.add_argument("--decay_step", type=int, default=11400)  #
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')  # 默认是true
    opt = parser.parse_args()
    return opt


def train(opt, train_loader, model, board):

    model.cuda()
    model.train()

    criterionL2 = nn.MSELoss()
    criterionL1 = nn.L1Loss(reduction='sum')
    criterionsmooth = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, threshold=)

    previous = 0
    epoch_iters = int((opt.keep_step + opt.decay_step) / opt.batch_size)
    for step in range(opt.keep_step + opt.decay_step):

        iter_start_time = time.time()
        inputs = train_loader.next_batch()
        image = inputs['image'].cuda(non_blocking=True).float()
        gt_rad = inputs['rad'].cuda(non_blocking=True).float()
        im_names = inputs['im_name']

        pred_rad, features = model(image)

        # loss = criterionL2(keypoints, target.float())
        gt_rad = gt_rad.view(-1, 1)

        rad = pred_rad.clone().detach().cpu().numpy()  # size: [B, 2]
        img = image.clone().detach().cpu().numpy()
        # array = np.transpose(np.zeros_like(img), axes=[0, 2, 3, 1])

        # assert array.shape[0] == opt.batch_size, 'generated temporary matrix shape is not matched '
        # for idx in range(image.size()[0]):
        #
        #     M = cv2.getRotationMatrix2D((opt.width / 2, opt.height / 2),
        #                                 angle=float(np.rad2deg(rad[idx, 0])), scale=1)
        #     rotated_image = cv2.warpAffine(np.transpose(img[idx, :, :, :],
        #                                 axes=[1, 2, 0]), M, (opt.width, opt.height))
        #     array[idx, :, :, :] = rotated_image

        # array_tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
        # visuals = [[image, array_tensor]]

        loss = criterionL2(pred_rad, gt_rad)

        optimizer.zero_grad()  # clear the old gradient
        loss.backward()  # compute the derivative of the loss with respect to the parameters
        optimizer.step()  # optimize the parameters
        scheduler.step()

        if (step+1) % opt.display_count == 0:
            board.add_scalar('train_loss', loss.item(), step + 1)
            # board_add_images(board, 'combine', visuals, step + 1)
            # board.add_scalar('lr', scheduler.get_lr(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, learning rate: %.6f, loss: %4f' % (
            step + 1, t, scheduler.get_lr()[0], loss.item()), flush=True)

        # if (step+1) % (epoch_iters*10) == 0:
        #
        #     model.eval()
        #     losses = AverageMeter()
        #     for i in range(57):
        #
        #         input = val_loader.next_batch()
        #
        #         img = input['image'].cuda()
        #         rad = input['rad'].cuda()
        #         im_names = input['im_name']
        #
        #         with torch.no_grad():
        #             # compute output
        #             output = model(img.float())
        #             val_loss = criterionL1((rad*180/3.1415), (output*180/3.1415))
        #
        #         losses.update(val_loss.item(), img.size(0))
        #
        #     board.add_scalar('val_loss', losses.sum, (step+1)/(epoch_iters*10))
        #     model.train()

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def main():

    opt = get_opt()
    print(opt)

    train_split = 'train'
    # val_split = 'val'
    # create dataset
    train_dataset = IdDataset(opt, train_split)
    # val_dataset = IdDataset(opt, val_split)
    # create dataloader
    train_loader = DataLoader(opt, train_dataset)
    # val_loader = DataLoader(opt, val_dataset)

    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))  # writer = SummaryWriter
    model = regression()

    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    train(opt, train_loader, model, board)
    save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'reg_final.pth'))


if __name__ == "__main__":
    main()