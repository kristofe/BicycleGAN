from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image


class HeightmapRenderingLoss(torch.nn.Module):
    # This generates a normal map from a heightmap using convolutions and is fully differentiable
    # TODO: Handle cuda calls
    def __init__(self, gpu_ids='', use_sobel=True):
        super(HeightmapRenderingLoss, self).__init__()

        self.gpu_ids = gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.last_normals = None
        self.last_generated_pixels = None
        self.last_target_pixels = None
        self.base_wts = self.get_blur_filters()
        self.pad = nn.ReplicationPad2d(1)  # basically forces a single sided finite diff at borders

    def get_blur_filters(self):
        c = 1.0/4.0
        wts = torch.FloatTensor([
            [
                [
                    [ 0, c,  0],
                    [c,  0 ,c],
                    [  0, c,  0],
                ]
            ]
        ])
        return wts

    def normals_to_diffuse_render(self, normals, ao):
        w = normals.size(3)
        h = normals.size(2)
        b = normals.size(0)
        ld1 = torch.FloatTensor([
            [0.7053],
            [-0.7053],
            [0.7053]
        ])  # normalized light dir from OGLViewer

        ld2 = torch.FloatTensor([
            [0.0],
            [0.0],
            [1.0]
        ])  # normalized light dir from OGLViewer

        ld1 = ld1.unsqueeze(1).unsqueeze(0)
        ld1 = ld1.expand(b, 3, h, w)
        ld2 = ld2.unsqueeze(1).unsqueeze(0)
        ld2 = ld2.expand(b, 3, h, w)
        dot1 = normals * ld1
        dot2 = normals * ld2
        d1_img = dot1[:,0:1,:,:] + dot1[:,1:2,:,:] + dot1[:,2:3,:,:]
        #d2_img = dot2[:,0:1,:,:] + dot2[:,1:2,:,:] + dot2[:,2:3,:,:]

        zero_values = torch.zeros(d1_img.size())
        d1_img = torch.max(d1_img, zero_values)
        #d2_img = torch.max(d2_img, zero_values)
        #diffuse_img = d1_img * 0.65 + d2_img * 0.3 + 0.05
        diffuse_img = d1_img * 0.3 + 0.7

        diffuse_img = diffuse_img * ao


        diffuse_img_int = diffuse_img * 255
        diffuse_img_int = diffuse_img_int[0,0:1,:,:]
        diffuse_img_int = diffuse_img_int.expand(3, h, w)
        diffuse_img_int = diffuse_img_int.permute(1,2,0)
        #im = Image.fromarray(diffuse_img_int.numpy().astype(np.uint8))
        return diffuse_img_int

    def simplistic_height_to_AO(self, x, wts, scale=4.0):
        assert(x.dim() == 4)  # assume its a batch of 2D images
        channels = x.size(1)
        assert(channels == 1)  # Assuming a 1 channel grayscale image
        w = x.size(3)
        h = x.size(2)
        b = x.size(0)

        p = nn.ReplicationPad2d(1) # basically forces a single sided finite diff at borders
        rx = x.clone()

        # TODO: Make this a faster blur!!!!!
        for i in range(32):
            rx = p(rx)
            rx = F.conv2d(rx, wts, bias=None, stride=1, padding=0)

        output = (rx - x) * scale
        output = 1.0 - torch.clamp(output, min=0, max=1.0)
        #im = output * 255
        #im = im.squeeze().expand(3, h, w)
        #im = im.permute(1,2,0)
        #img = Image.fromarray(im.numpy().astype(np.uint8))
        #img.show()
        return output

    @staticmethod
    def render_tensor2im(t):
        # assumes 1 x 3 x W x H tensor
        t = t * 255
        print("render_tensor2im input size: {}".format(t.size()))
        t = t.squeeze().expand(3, t.size(1), t.size(2)).permute(1,2,0)
        return t.detach().cpu().float().numpy().astype(np.uint8)

    @staticmethod
    def adjust_filters_to_batchsize(batchsize, base_wts):
        # no memory should be allocated here... just new views are created
        x_wts = base_wts.expand(batchsize, 1, 3, 3)
        return x_wts

    def convert_render_to_image(self, render):
        assert(render is not None)
        #imgs = []
        #for i in range(normals.size(0)):
        #    img = self.normals_to_image(normals[i])
        #    imgs.append(img)
        #return img
        return self.render_tensor2im(render[0])

    def forward(self, *x):
        generated_height_data, normals = x[0], x[1]
        self.last_normals = normals
        b = generated_height_data.size(0)

        if generated_height_data.size(1) != 1:
            generated_height_data = generated_height_data[:,0:1,:,:]

        wts = HeightmapRenderingLoss.adjust_filters_to_batchsize(b, self.base_wts)
        ao = self.simplistic_height_to_AO(generated_height_data, wts)
        self.last_generated_pixels = self.normals_to_diffuse_render(normals, ao)
        return self.last_generated_pixels

    '''
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
    '''
