import os
import os.path as osp
import numpy as np
import json
import glob
import logging

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
import cv2
import math
import matplotlib.pyplot as plt
from prefetch_generator import BackgroundGenerator
# from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class IdDataset(data.Dataset):

    def __init__(self, opt, datamode):
        super(IdDataset, self).__init__()
        self.opt = opt
        self.data_root = opt.dataroot
        self.image_width = opt.width
        self.image_height = opt.height
        self.datamode = datamode
        self.num_joints = 10
        # self.max_rotation = 8

        self.image_list = []
        self.target = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.transform_single = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        self.translate = transforms.Compose([
            transforms.RandomAffine(degrees=[0, 0], translate=[0.1, 0.]),
            transforms.ColorJitter(brightness=0, contrast=(0.7, 1.2), saturation=0, hue=0)
        ])

        for name in os.listdir(osp.join(self.data_root, self.datamode)):

            prefix, extension = os.path.split(osp.join(self.data_root, self.datamode, name))
            if extension.endswith('png'):
                image_name = osp.join(self.data_root, self.datamode, name)
                txt_file = name.split('.')[0] + '.txt'
                self.image_list.append(image_name)
                self.target.append(txt_file)

        # for name in glob.glob(osp.join(self.data_root, self.datamode, '*.json')):
        #     # print('processing:', name)
        #     with open(name, encoding='utf-8') as f:
        #         label_file = json.load(f)
        #         image_name = label_file["imagePath"]
        #         img_name = image_name.split('.')[0] + '_torso' + '.' + image_name.split('.')[1]
        #         image_path = osp.join(self.data_root, self.datamode, img_name)
        #         self.image_list.append(image_path)
        #         upper_cord = []
        #         down_cord = []
        #         left_cord = []
        #         right_cord = []
        #         for i in range(4):
        #
        #             if label_file['shapes'][i]['label'] == '上轴':
        #                 upper_cord.append(label_file['shapes'][i]['points'])
        #
        #             elif label_file['shapes'][i]['label'] == '下轴':
        #                 down_cord.append(label_file['shapes'][i]['points'])
        #
        #             elif label_file['shapes'][i]['label'] == '左肩':
        #                 left_cord.append(label_file['shapes'][i]['points'])
        #
        #             elif label_file['shapes'][i]['label'] == '右肩':
        #                 right_cord.append(label_file['shapes'][i]['points'])
        #
        #         tmp1 = [item for sublist in upper_cord[0] for item in sublist]
        #         tmp2 = [item for sublist in down_cord[0] for item in sublist]
        #         tmp3 = [item for sublist in left_cord[0] for item in sublist]
        #         tmp4 = [item for sublist in right_cord[0] for item in sublist]
        #
        #         keypoints = tmp1 + tmp2 + tmp3 + tmp4
        #         self.target.append(keypoints)

        self.num_images = len(self.image_list)
        logger.info('=> num_images: {}'.format(self.num_images))
        # print(self.image_list)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):

        image_name = self.image_list[index]
        # print('processing:', image_name)
        # keypoints = self.target[index]
        target_path = self.target[index]
        with open(osp.join(self.data_root, self.datamode, target_path), 'r') as f:
            angle = f.readline()

        original_image = cv2.imread(image_name)
        if original_image is None:
             raise ValueError('Fail to read {}'.format(image_name))
        h, w, c = original_image.shape
        # image, keypoints, pad_up, pad_bottom, pad_left, pad_right, ratio = \
        #     self._apply_resize_padding(original_image, h, w, keypoints, image_name)
        # keypoints, rad = self.get_radius(keypoints)

        # aug_rot = np.rad2deg(rad)
        # M = cv2.getRotationMatrix2D((self.image_width / 2, self.image_height / 2), aug_rot,
        #                             1)  # Positive values mean counter-clockwise rotation
        # dst = cv2.warpAffine(image, M, (self.image_width, self.image_height))

        # plt.figure(1)
        # plt.subplot(121)
        # plt.imshow(image[:, :, ::-1])
        # plt.subplot(122)
        # plt.imshow(dst[:, :, ::-1])
        # plt.show()

        # keypoints = self.normalization(keypoints)  # normalize the coordinate to [0, 1]
        # convert ndarray to Tensors
        if self.datamode == 'train':
            prob = random.random()
            if prob > 0.5:
                original_image = original_image[:, :, ::-1]
                angle = -float(angle)

        image = self.transform(original_image)
        # keypoints = torch.from_numpy(np.array(keypoints))
        rad = torch.from_numpy(np.array(float(angle)))

        result = {
            'image': image,
            'im_name': os.path.basename(image_name),
            # 'keypoints': keypoints,
            'rad': rad,
            # 'pad_up': pad_up,
            # 'pad_bottom': pad_bottom,
            # 'pad_left': pad_left,
            # 'pad_right': pad_right,
            # 'ratio': ratio
        }

        return result

    def _apply_resize_padding(self, img, h, w, keypoints, image_name):

        # prefix, extension = os.path.splitext(image_name)
        # alpha_name = image_name.split('\\')[-1].split('.')[0] + '-src-mask' + '.' + extension
        # alpha = cv2.imread(osp.join(prefix, image_name))
        center_x, center_y = self.get_keypoints_center(keypoints)
        kpts = keypoints.copy()

        if h > w:
            # resize the image to a fixed size
            ratio = self.image_height / h
            img = cv2.resize(img, dsize=(int(ratio*w), int(ratio*h)), interpolation=cv2.INTER_CUBIC)
            # map = cv2.resize(map, dsize=(int(ratio*w), int(ratio*h)), interpolation=cv2.INTER_CUBIC)
            pad_bottom = self.image_height - int(ratio*h)
            pad_left = (self.image_width - int(ratio*w)) // 2
            pad_right = (self.image_width -int(ratio*w)) - (self.image_width - int(ratio*w)) // 2
            pad_up = 0
            img = np.pad(img, ((0, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0)

            # calculate the refreshed coordinate
            kpts_vec = []
            for i in range(20):
                if i % 2 == 0:
                   kpts_vec.append(kpts[i] - center_x)
                else:
                   kpts_vec.append(kpts[i] - center_y)
            kpts_vec = kpts_vec * np.array(ratio)
            center_x = center_x * np.array(ratio)
            center_y = center_y * np.array(ratio)

            for i in range(20):
                if i % 2 == 0:
                    kpts[i] = kpts_vec[i] + center_x
                else:
                    kpts[i] = kpts_vec[i] + center_y

            for i in range(20):
                if i % 2 == 0:
                    kpts[i] = kpts[i] + pad_left
                else:
                    pass

            return img, kpts, pad_left, pad_right, pad_bottom, pad_up, ratio

            # if self.datamode == 'train':
            #     aug_rot = (np.random.random() * 2 - 1) * self.max_rotation
            #     M = cv2.getRotationMatrix2D((self.image_width / 2, self.image_height / 2), -aug_rot, 1)  #  Positive values mean counter-clockwise rotation
            #     dst = cv2.warpAffine(img, M, (self.image_width, self.image_height))
            #     angle_radian = aug_rot * math.pi / 180
            #     x_center = self.image_width/2
            #     y_center = self.image_height/2
            #     up_x = int(math.cos(-angle_radian) * (kpts[4] - x_center) + math.sin(-angle_radian) * (kpts[5] - y_center) + x_center + 0.5)
            #     up_y = int(-math.sin(-angle_radian) * (kpts[4] - x_center) + math.cos(-angle_radian) * (kpts[5] - y_center) + y_center + 0.5)
            #     up_x = min(self.image_width - 1, max(0, up_x))
            #     up_y = min(self.image_height - 1, max(0, up_y))
            #
            #     kpts[4] = up_x
            #     kpts[5] = up_y
            #
            #     down_x = int(math.cos(-angle_radian) * (kpts[6] - x_center) + math.sin(-angle_radian) * (kpts[7] - y_center) + x_center + 0.5)
            #     down_y = int(-math.sin(-angle_radian) * (kpts[6] - x_center) + math.cos(-angle_radian) * (kpts[7] - y_center) + y_center + 0.5)
            #     down_x = min(self.image_width - 1, max(0, down_x))
            #     down_y = min(self.image_height - 1, max(0, down_y))
            #
            #     kpts[6] = down_x
            #     kpts[7] = down_y
            #
            #     return dst, kpts, pad_left, pad_right, pad_bottom, pad_up, ratio
            #
            # else:
            #
            #     return img, kpts, pad_left, pad_right, pad_bottom, pad_up, ratio

        else:
            ratio = self.image_width / w
            img = cv2.resize(img, dsize=(int(ratio*w), int(ratio*h)), interpolation=cv2.INTER_CUBIC)
            # map = cv2.resize(map, dsize=(int(ratio*w), int(ratio*h)), interpolation=cv2.INTER_CUBIC)
            pad_right = self.image_width - int(ratio * w)
            pad_up = (self.image_height - int(ratio * h)) // 2
            pad_bottom = (self.image_height - int(ratio * h)) - (self.image_height - int(ratio * h)) // 2
            img = np.pad(img, ((pad_up, pad_bottom), (0, pad_right), (0, 0)), 'constant', constant_values=0)
            # map = np.pad(map, ((pad_up, pad_bottom), (0, pad_right)), 'constant', constant_values=0)
            pad_left = 0

            # calculate the refreshed coordinate
            kpts_vec = []
            for i in range(20):
                if i % 2 == 0:
                    kpts_vec.append(kpts[i] - center_x)
                else:
                    kpts_vec.append(kpts[i] - center_y)
            kpts_vec = kpts_vec * np.array(ratio)
            center_x = center_x * np.array(ratio)
            center_y = center_y * np.array(ratio)

            for i in range(20):
                if i % 2 == 0:
                    kpts[i] = kpts_vec[i] + center_x
                else:
                    kpts[i] = kpts_vec[i] + center_y

            for i in range(20):
                if i % 2 == 0:
                    pass
                else:
                    kpts[i] = kpts[i] + pad_up

            return img, kpts, pad_left, pad_right, pad_bottom, pad_up, ratio

            # if self.datamode == 'train':
            #     aug_rot = (np.random.random() * 2 - 1) * self.max_rotation
            #     M = cv2.getRotationMatrix2D((self.image_width / 2, self.image_height / 2), -aug_rot, 1)
            #     dst = cv2.warpAffine(img, M, (self.image_width, self.image_height))
            #     angle_radian = aug_rot * math.pi / 180
            #     x_center = self.image_width / 2
            #     y_center = self.image_height / 2
            #     up_x = int(math.cos(-angle_radian) * (kpts[4] - x_center) + math.sin(-angle_radian) * (
            #                 kpts[5] - y_center) + x_center + 0.5)
            #     up_y = int(-math.sin(-angle_radian) * (kpts[4] - x_center) + math.cos(-angle_radian) * (
            #                 kpts[5] - y_center) + y_center + 0.5)
            #     up_x = min(self.image_width - 1, max(0, up_x))
            #     up_y = min(self.image_height - 1, max(0, up_y))
            #
            #     kpts[4] = up_x
            #     kpts[5] = up_y
            #
            #     down_x = int(math.cos(-angle_radian) * (kpts[6] - x_center) + math.sin(-angle_radian) * (
            #                 kpts[7] - y_center) + x_center + 0.5)
            #     down_y = int(-math.sin(-angle_radian) * (kpts[6] - x_center) + math.cos(-angle_radian) * (
            #                 kpts[7] - y_center) + y_center + 0.5)
            #     down_x = min(self.image_width - 1, max(0, down_x))
            #     down_y = min(self.image_height - 1, max(0, down_y))
            #
            #     kpts[6] = down_x
            #     kpts[7] = down_y
            #
            #     return dst, kpts, pad_left, pad_right, pad_bottom, pad_up, ratio
            #
            # else:
            #
            #     return img, kpts, pad_left, pad_right, pad_bottom, pad_up, ratio

    def get_keypoints_center(self, keypoints):

        x = keypoints[0::2]
        y = keypoints[1::2]
        x_center = (min(x[:])+max(x[:])) / 2
        y_center = (min(y[:])+max(y[:])) / 2

        return [x_center, y_center]

    def get_radius(self, keypoints):

        # formulation: radians = degrees \cdot (pi/180)
        kpts = keypoints.copy()
        rad = np.arctan2(kpts[7] - kpts[5], kpts[6] - kpts[4]) - np.pi/2

        kpts.pop(6)
        kpts.pop(6)
        kpts.insert(len(kpts), rad)
        rad = kpts[18]
        # print(len(kpts))
        # assert len(kpts) == 19, 'the total length does not equal 9 points plus one radius to be learned'

        return kpts, rad

    def normalization(self, keypoints):

        kpts = keypoints.copy()
        for idx in range(len(kpts)-1):
            if idx % 2 == 0:
                kpts[idx] = kpts[idx] / self.image_height
            else:
                kpts[idx] = kpts[idx] / self.image_width

        return kpts

    def _apply_rotate(self):
        """rotate image in a range"""
        raise NotImplementedError


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = DataLoaderX(dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                 num_workers=opt.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        # self.data_loader = torch.utils.data.DataLoader(
        #     dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        #     num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for pose estimation!")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default=r"G:\KeypointsRegression\dataset")  # '/home/vampire/WorkSpace/nick/cp-vton-id/data'
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--radius", type=int, default=10)  # 半径范围
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument("--datamode", default="train")

    opt = parser.parse_args()
    dataset = IdDataset(opt, datamode='train')
    data_loader = DataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d'
          % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    # for i in range(165):
    #     print('processing:', i)
    #     first_item = dataset.__getitem__(i)
    #     image = first_item['image']

        # keypoints = first_item['keypoints']
        # pad_up = first_item['pad_up']
        # pad_bottom = first_item['pad_bottom']
        # pad_left = first_item['pad_left']
        # pad_right = first_item['pad_right']
        # ratio = first_item['ratio']

        # show the image
        # x = []
        # y = []
        # for i in range(18):
        #     if i % 2 == 0:
        #         x.append(int(keypoints[i]*512))
        #     else:
        #         y.append(int(keypoints[i]*512))
        # plt.figure()
        # plt.imshow(image)
        # plt.scatter(x, y, s=3, c='r')
        # plt.show()

    first_batch = data_loader.next_batch()
    from IPython import embed;
    embed()