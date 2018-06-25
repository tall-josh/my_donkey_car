import os
import numpy as np
import matplotlib.image as matimg
import json
import csv

'''load a single image'''
def load_image(path):
    img = matimg.imread(path)
    return img

'''load a single annotation'''
def load_anno(path):
    with open(path, 'r') as f:
        anno = json.load(f)
    return anno

'''Load all images and annotations from list of paths'''
def load_image_anno_pair(image_dir, anno_dir, name):
    im_path = os.path.join(image_dir, f"{name}.jpg")
    an_path = os.path.join(anno_dir, f"{name}.json")
    return load_image(im_path), load_anno(an_path)

'''
    Loads image and annotatios from data_set_path
'''
def load_dataset(data_set_path):
    data = []
    with open(data_set_path, 'r') as data_set:
        data_reader = csv.reader(data_set, delimiter=',')
        for name in data_reader:
            data.append(name[0])
    return data


'''
- Bin the steering angles.
  bin_number = data_value // bin_size
  ie: 0 to 14 with 5 bins is 3 elements per bin
'''
def bin_value(value, num_bins, val_range=1024):
    bin_size = val_range / num_bins
    return value // bin_size

'''
 Magic numbers to convert RGB to gray scale
'''
def rgb2gray(data):
    gray = []
    for rgb in data:
        temp = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        temp = np.expand_dims(temp, axis=2)
        gray.append(temp)
    gray = np.asarray(gray)
    return gray
