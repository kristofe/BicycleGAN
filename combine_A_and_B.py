from pdb import set_trace as st
import os
import numpy as np
#import cv2
from PIL import Image
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='./datasets/terrain_2560/trainA')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='./datasets/terrain_2560/trainB')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='./datasets/terrain_2560/train')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_input, 0001_target_uint8) to (0001_AB)',action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

args.use_AB = True
img_fold_A = os.listdir(args.fold_A)
img_fold_B = os.listdir(args.fold_B)

# this is for subdirectories i guess
img_list = img_fold_A
if args.use_AB:
    img_list = [img_path for img_path in img_list if '_input.' in img_path]

num_imgs = min(args.num_imgs, len(img_list))
print('use %d/%d images' % ( num_imgs, len(img_list)))
img_fold_AB = args.fold_AB
if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)
print('number of images = %d' % (num_imgs))
for n in range(num_imgs):
    name_A = img_list[n]
    path_A = os.path.join(args.fold_A, name_A)
    if args.use_AB:
        name_B = name_A.replace('_input.', '_target_uint8.')
    else:
        name_B = name_A
    path_B = os.path.join(args.fold_B, name_B)
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        name_AB = name_A
        if args.use_AB:
            name_AB = name_AB.replace('_input.', '.') # remove _input
        path_AB = os.path.join(img_fold_AB, name_AB)
        print path_A, path_B
        im_A = np.array(Image.open(path_A))
        im_B = np.array(Image.open(path_B))

        im_B = np.expand_dims(im_B, axis=2)
        im_B = np.repeat(im_B, 3, axis=2)

        im_AB = np.concatenate([im_A, im_B], 1)
        save_AB = Image.fromarray(im_AB)
        save_AB.save(path_AB)

