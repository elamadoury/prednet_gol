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


usage = 'Usage: python {} SETTINGS_FILE [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate .hkl files for train and test images',
                                 usage=usage)
parser.add_argument('settings_file', action='store', nargs=None, 
                    type=str, help='dataset settings file with data path.')
args = parser.parse_args()

settings_file = args.settings_file.replace('.py', '').replace('/', '.')
_, module = settings_file.rsplit('.', 1)
settings = __import__(settings_file, fromlist="DATA_DIR")
DATA_DIR = settings.DATA_DIR

desired_im_sz = (128, 160)

# Processes images and saves them in train, val, test splits.
def process_data():
    #DATA_DIR = args.settings_file
    im_dir = settings.DATA_DIR + "/images/"
    split = "test"
    im_list = []
    source_list = []  # corresponds to recording that image came from
    for i in range(36378,(36378+50)): 
        image_name = "frame_" + str(i).zfill(5)  + ".jpg" 
        im_list += [im_dir + image_name]
        source_list += [im_dir]

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


#store images in a reverse order
def reverse_process_data():
    im_dir = DATA_DIR + "/images/"
    print("DATA_DIR", DATA_DIR)
    split = "test"
    im_list = []
    source_list = []  # corresponds to recording that image came from
    for i in reversed(range(36378,(36378+50))): 
        image_name = "frame_" + str(i).zfill(5)  + ".jpg" 
        im_list += [im_dir + image_name]
        source_list += [im_dir]

    print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
    X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
    for i, im_file in enumerate(im_list):
        im = imread(im_file)
        X[i] = process_im(im, desired_im_sz)

    hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
    hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))

if __name__ == '__main__':
    #process_data()
    reverse_process_data()
