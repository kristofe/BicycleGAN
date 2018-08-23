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
    def __init__(self, gpu_ids='', use_sobel=True):
        super(HeightmapNormalsLoss, self).__init__()

        self.gpu_ids = gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.use_sobel = use_sobel
        self.bias = None
        self.last_generated_normals = None
        self.last_target_normals = None
        if self.use_sobel:
            self.base_x_wts, self.base_y_wts = self.get_sobel_filters()
        else:
            self.base_x_wts, self.base_y_wts = self.get_simple_filters()
        self.pad = nn.ReplicationPad2d(1)  # basically forces a single sided finite diff at borders
        #self.loss = torch.nn.L1Loss()

    def get_sobel_filters(self):
        x_wts = self.Tensor([
            [
                [
                    [1.0, 0.0, -1.0],
                    [2.0, 0.0, -2.0],
                    [1.0, 0.0, -1.0],
                ]
            ]
        ])
        y_wts = self.Tensor([
            [
                [
                    [1.0,  2.0,  1.0],
                    [0.0,  0.0,  0.0],
                    [-1.0, -2.0, -1.0],
                ]
            ]
        ])
        return x_wts, y_wts


    def get_simple_filters(self):
        x_wts = self.Tensor([
            [
                [
                    [0.0, 0.0,  0.0],
                    [1.0, 0.0, -1.0],
                    [0.0, 0.0,  0.0],
                ]
            ]
        ])
        y_wts = self.Tensor([
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
        #return Image.fromarray(n.detach().cpu().float().numpy().astype(np.uint8))
        return n.detach().cpu().float().numpy().astype(np.uint8)

    def convert_normals_to_image(self, normals):
        assert(normals is not None)
        #imgs = []
        #for i in range(normals.size(0)):
        #    img = self.normals_to_image(normals[i])
        #    imgs.append(img)
        #return img
        return self.normals_to_image(normals[0])

    def calculate_normals(self, x):
        assert(x.dim() == 4)  # assume its a batch of 2D images
        batchsize = x.size(0)
        channels = x.size(1)
        assert(channels == 1)  # Assuming a 1 channel grayscale image

        x_wts, y_wts = self.adjust_filters_to_batchsize(batchsize, self.base_x_wts, self.base_y_wts)

        x = self.pad(x)
        gx = F.conv2d(x, x_wts, bias=None, stride=1, padding=0)
        gy = F.conv2d(x, y_wts, bias=None, stride=1, padding=0)

        # prevent nan's by clamping so gz_start is never >= 1.0
        # that would cause taking sqrt of 1.0 - gz_start to be nan.
        gz_start = gx * gx - gy * gy
        gz_start = torch.clamp(gz_start, -1.0, 0.99999)
        gz_start = 1.0 - gz_start

        if torch.min(gz_start) < 0.0:
            print(gz_start[gz_start < 0.0].size())
            gz_start[gz_start < 0.0] = 0.000001

        # the leading coefficient controls sharpness.
        # Default should be 0.5.
        # < 1.0 is sharper.
        # > 1.0 is smoother
        gz = 0.25 * (gz_start).sqrt()
        #gz = 0.25 * (1.0 - gx * gx - gy * gy).sqrt()

        norm = torch.cat((gx, gy, gz), 1)

        gx = 2.0 * gx
        gy = 2.0 * gy
        length = (gx*gx + gy*gy + gz*gz).sqrt()
        normals = norm/length
        if torch.isnan(normals).any():
            print("There are nan normals")

        return normals

    def forward(self, *x):
        generated_height_data = x[0]
        if generated_height_data.size(1) != 1:
            generated_height_data = generated_height_data[:,0:1,:,:]

        #target_height_data = x[1]
        self.last_generated_normals = self.calculate_normals(generated_height_data)
        #self.last_target_normals = self.calculate_normals(target_height_data)

        #return self.loss(self.last_generated_normals, self.last_target_normals)
        return self.last_generated_normals
