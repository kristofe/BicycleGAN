from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image


class HeightmapNormalsLoss(torch.nn.Module):
    # This generates a normal map from a heightmap using convolutions and is fully differentiable
    # TODO: Handle cuda calls
    def __init__(self, use_sobel=True):
        super(HeightmapNormalsLoss, self).__init__()

        self.use_sobel = use_sobel
        self.bias = None
        self.last_generated_normals = None
        self.last_target_normals = None
        if self.use_sobel:
            self.base_x_wts, self.base_y_wts = self.get_sobel_filters()
        else:
            self.base_x_wts, self.base_y_wts = self.get_simple_filters()
        self.loss = torch.nn.L1Loss()

    @staticmethod
    def get_sobel_filters():
        x_wts = torch.FloatTensor([
            [
                [
                    [1.0, 0.0, -1.0],
                    [2.0, 0.0, -2.0],
                    [1.0, 0.0, -1.0],
                ]
            ]
        ])
        y_wts = torch.FloatTensor([
            [
                [
                    [1.0,  2.0,  1.0],
                    [0.0,  0.0,  0.0],
                    [-1.0, -2.0, -1.0],
                ]
            ]
        ])
        return x_wts, y_wts

    @staticmethod
    def get_simple_filters():
        x_wts = torch.FloatTensor([
            [
                [
                    [0.0, 0.0,  0.0],
                    [1.0, 0.0, -1.0],
                    [0.0, 0.0,  0.0],
                ]
            ]
        ])
        y_wts = torch.FloatTensor([
            [
                [
                    [0.0,  1.0,  0.0],
                    [0.0,  0.0,  0.0],
                    [0.0, -1.0,  0.0],
                ]
            ]
        ])
        return x_wts, y_wts

    @staticmethod
    def adjust_filters_to_batchsize(batchsize, base_x_wts, base_y_wts):
        # no memory should be allocated here... just new views are created
        x_wts = base_x_wts.expand(batchsize, 1, 3, 3)
        y_wts = base_y_wts.expand(batchsize, 1, 3, 3)
        return x_wts, y_wts

    @staticmethod
    def normals_to_image(n):
        n = (n * 0.5 + 0.5) * 255
        # assumes 1 x 3 x W x H tensor
        n = n.squeeze().permute(1, 2, 0)
        return Image.fromarray(n.numpy().astype(np.uint8))

    def get_images_from_last_normals(self, normals):
        assert(normals is not None)
        imgs = []
        for i in range(normals.size(0)):
            img = self.normals_to_image(normals[i])
            imgs.append(img)
        return imgs

    def calculate_normals(self, x):
        assert(x.dim() == 4)  # assume its a batch of 2D images
        batchsize = x.size(0)
        channels = x.size(1)
        assert(channels == 1)  # Assuming a 1 channel grayscale image

        x_wts, y_wts = self.adjust_filters_to_batchsize(batchsize, self.base_x_wts, self.base_y_wts)

        p = nn.ReplicationPad2d(1)  # basically forces a single sided finite diff at borders
        x = p(x)
        gx = F.conv2d(x, x_wts, bias=None, stride=1, padding=0)
        gy = F.conv2d(x, y_wts, bias=None, stride=1, padding=0)

        # the leading coefficient controls sharpness.
        # Default should be 0.5.
        # < 1.0 is sharper.
        # > 1.0 is smoother
        gz = 0.25 * (1.0 - gx * gx - gy * gy).sqrt()

        norm = torch.cat((gx, gy, gz), 1)

        gx = 2.0 * gx
        gy = 2.0 * gy
        length = (gx*gx + gy*gy + gz*gz).sqrt()
        return norm/length

    def forward(self, *x):
        generated_height_data = x[0]
        target_height_data = x[1]
        self.last_generated_normals = self.calculate_normals(generated_height_data)
        self.last_target_normals = self.calculate_normals(target_height_data)

        return self.loss(self.last_generated_normals, self.last_target_normals)
