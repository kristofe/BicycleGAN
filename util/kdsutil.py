import math
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.utils.data
from torchvision import datasets, transforms, utils
from PIL import Image

def features2gridim(feature_tensor, count=32, rows=4, normalize=True, scale_each=False):
  ft = feature_tensor.detach().cpu().float().permute(1,0,2,3)
  im = utils.make_grid(ft[0:count], nrow=rows, normalize=normalize, scale_each=scale_each)
  im = im.unsqueeze(0)
  return im

def visualize_gradients(viz, data, caption='', zoom=4):
  batchSize = data.size(0)
  rows = int(math.sqrt(batchSize))
  toPIL = transforms.ToPILImage()
  # normalize it
  data = data.cpu()
  dmin = data.min()
  dmax = data.max()
  width = dmax - dmin
  if (width > 0.0):
    data = data.add(-dmin).div(width)

  data_imgs = utils.make_grid(data, nrow=rows)
  pimg = toPIL(data_imgs)
  pimg = pimg.resize((pimg.height * zoom, pimg.width * zoom), Image.NEAREST)
  imgarray = np.array(pimg)
  new_image = torch.from_numpy(imgarray)
  assert (new_image.dim() == 3)
  # new_image = new_image.permute(2,0,1)
  #viz.showImage(new_image, caption)


def visualize_model_gradients(viz, pmodel, epoch=0):
  for key, value in list(pmodel._modules.items()):
    if isinstance(value, nn.Conv2d) or isinstance(value, nn.ConvTranspose2d):
      # Get the first 64 gradients if there are that many
      sizes = value.weight.data.size()
      data = value.weight.data.view(sizes[0] * sizes[1], 1, sizes[2], sizes[3])
      maps = min(sizes[0] * sizes[1], 64)
      data = data[0:maps, :, :, :]
      visualize_gradients(viz,data, "epoch {:d} {} weights".format(epoch, key), 8)
      sizes = value.weight.grad.data.size()
      data = value.weight.grad.data.view(sizes[0] * sizes[1], 1, sizes[2],
                                         sizes[3])
      maps = min(sizes[0] * sizes[1], 64)
      data = data[0:maps, :, :, :]
      visualize_gradients(viz, data, "epoch {:d} {} grad".format(epoch, key), 8)


def visualize_batch(viz, data, caption='', normalize=True, gammacorrect=False,
                    window=None):
  if(gammacorrect):
    gamma = 2.20
    data = data.pow(1.0/gamma)
  if(normalize == False):
    #data = data.mul(0.5).add(0.5).clamp(0, 1)
    data = data.clamp(0, 1)
  else:
    dmin = data.min()
    dmax = data.max()
    width = dmax - dmin
    if (width > 0.0):
      data = data.add(-dmin).div(width)

  #data_imgs = utils.make_grid(data).permute(1, 2, 0)
  data_imgs = utils.make_grid(data)

  #viz.showImage(data_imgs, caption, window=window)