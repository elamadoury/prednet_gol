# Create image datasets.

import argparse
import importlib
import numpy as np
import os
import sys
import requests
import urllib.request
import sys

from bs4 import BeautifulSoup
import hickle as hkl
from imageio import imread
from scipy.misc import imresize


usage = 'Usage: python {} DATA_DIR [N_IMAGES] [ORDER] [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate .hkl files for train, test and val images',
                                 usage=usage)
parser.add_argument('data_dir', action='store', nargs=None, 
                    type=str, help='path to directory containing image folder.')
parser.add_argument('n_images', action='store', nargs='?', default=-1,
                    type=int, help='optional: total number of images to use.')
parser.add_argument('order', action='store', nargs='?', default=0,
                    type=int, help='optional: 0 for regular order, 1 for inverse ordering of frames.')
args = parser.parse_args()

DATA_DIR = args.data_dir
desired_im_sz = (128, 160)

#train, val, test
split_ratio =[0.8,0.1,0.1]
splits = ["train", "test", "val"]

# Processes images and saves them in train, val, test splits.
# Order : 0 for normal, 1 for reverse
def process_data(n_images=-1, order=0):
    im_dir = DATA_DIR + "/images/"
    if n_images==-1:
        n_images = len([name for name in os.listdir(im_dir) if os.path.isfile(os.path.join(im_dir, name))])

    if order == 0:
        start = 0
    else:
        start = n_images

    for s in range(len(splits)):
        split = splits[s]
        im_list = []
        source_list = []  # corresponds to recording that image came from (for kitti)

        if order == 0:
            frame_range = range(start,int(start+split_ratio[s]*n_images))
        else:
            frame_range = reversed(range(int(start-split_ratio[s]*n_images), start))

        for i in frame_range: 
            image_name = "frame_" + str(i).zfill(5)  + ".jpg" 
            im_list += [im_dir + image_name]
            source_list += [im_dir]

        if order == 0:
            start += int(split_ratio[s]*n_images)
        else:
            start -= int(split_ratio[s]*n_images)

        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))

# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im

if __name__ == '__main__':
    process_data(args.n_images, args.order)
