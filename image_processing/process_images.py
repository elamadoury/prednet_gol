# Create image datasets.


import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl


usage = 'Usage: python {} SETTINGS_FILE [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate .hkl files for train and test.',
                                 usage=usage)
parser.add_argument('settings_file', action='store', nargs=None, 
                    type=str, help='dataset settings file with data path.')
args = parser.parse_args()

desired_im_sz = (128, 160)

from args.settings_file import *

# Processes images and saves them in train, val, test splits.
def process_data():
    #DATA_DIR = args.settings_file
    im_dir = DATA_DIR + "images/"
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
    im_dir = DATA_DIR + "images/"
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
