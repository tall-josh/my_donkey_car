import numpy as np
import csv
import io
from utils import *
from tqdm import tqdm
import common

NUM_BINS = common.NUM_BINS

# -----------------------------------------------------------------------------
#                   Some preprocess callback functions
#                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pass one of these functions (or make your own) to the 'preprocess_fn'
# argument of DataGenerator.
# -----------------------------------------------------------------------------

"""
- Normalize image
- Normalize annotation["throttle"]
- Bins annotation["steering"]
"""
def preprocess_normalize_images_bin_annos(im, an):
    num_bins  = NUM_BINS
    im = normalize_image(np.array(im))
    an["steering"] = bin_value(an["steering"],
                               num_bins,
                               val_range=1024)
    an["throttle"] = an["throttle"] / 1024
    pair = {}
    pair["original_image"]  = im
    pair["anno"]            = an
    return pair

"""
- Normalize image
- Leaves annotations untouched
- Returns dict of 'original_image' and 'anno'
"""
def preprocess_normalize_images_only(im, an):
    im = normalize_image(np.array(im))
    pair = {}
    pair["original_image"]  = im
    pair["anno"]            = an
    return pair

# -----------------------------------------------------------------------------
#                     Some prepare callback functions
#                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pass one of these functions (or make your own) to the 'data_aug_fn'
# argument of DataGenerator.
# -----------------------------------------------------------------------------

""" Some prepare callback functions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     Pass one of these functions (or make your own) to the 'data_aug_fn'
     argument of DataGenerator.
"""
def batch_data_aug(gen_ref, marker, mirror=True):
    batch = None
    if mirror:
        batch = _aug_batch_images_and_labels_NO_MIRROR(gen_ref, marker)
    else:
        batch = _aug_batch_images_and_labels_RAND_MIRROR(gen_ref, marker)
    return batch

"""
gen_ref: Reference to DataGenerator
marker: Keeps track of the generator idx
"""
def _aug_batch_images_and_labels_NO_MIRROR(gen_ref, marker):
    num_bins = NUM_BINS
    batch = {'images'      : [],
             'annotations' : [],
             'names'       : []}
    for ele in range(gen_ref.batch_size):
        pair     = gen_ref.data[gen_ref._indexes[marker+ele]]
        im, an = pair["original_image"], pair["anno"]
        batch["images"].append(im)
        batch["annotations"].append(an)
        batch["names"].append(pair["name"])

 # if gray scale add single channel dim
    if len(np.shape(batch["images"])) == 3:
        batch["images"] = np.expand_dims(batch["images"], axis=3)
    return batch

"""
gen_ref: Reference to DataGenerator
marker: Keeps track of the generator idx
"""
def _aug_batch_images_and_labels_RAND_MIRROR(gen_ref, marker):
    num_bins = NUM_BINS
    batch = {'images'      : [],
             'annotations' : [],
             'names'       : []}
    for ele in range(gen_ref.batch_size):
        pair     = gen_ref.data[gen_ref._indexes[marker+ele]]
        im, an = pair["original_image"], pair["anno"]
        im, an = _mirror_at_random(im, an, num_bins)
        batch["images"].append(im)
        batch["annotations"].append(an)
        batch["names"].append(pair["name"])

 # if gray scale add single channel dim
    if len(np.shape(batch["images"])) == 3:
        batch["images"] = np.expand_dims(batch["images"], axis=3)
    return batch

# -----------------------------------------------------------------------------
#                             Helper Functions
# -----------------------------------------------------------------------------
"""
Mirror randomly, add some noise.
"""
def _mirror_at_random(image, anno, num_bins):
    # Create copy so not to modify the original data
    anno_aug = anno.copy()
    bin_original = anno_aug["steering"]
    if random.uniform(0.,1.) < 0.5:
        image = np.flip(image, 1)
        anno_aug["steering"] = (num_bins-1)-bin_original

#   TODO: AUGMENT
    return image, anno_aug

"""
Normalize, and zero mean data.
"""
def normalize_image(image, zero_mean=False):
        if zero_mean:
            image = ((image / 255.0) - 0.5) * 2.
        else:
            image = (image / 255.0)
        return image

class DataGenerator(object):
    """
    batch_size: number of samples passed to model at each step.
    dataset:    list of sample names.
    image_dir:  path to directory containing images.
    anno_dir:   path to annotation directory.
    shuffle:    to shuffle or not to shuffle, that is the question. Setting to
                False is good for doing validation on a sequence of frames.
    preprocess_fn   : callback function applied to images prior to passing into
                      network.
    data_aug_fn: any other processing needed on a batch by batch basis.
    """
    def __init__(self, batch_size, dataset, image_dir, anno_dir, shuffle=True,
                 preprocess_fn=None, data_aug_fn=None, random_mirror=True):

        self.preprocess      = preprocess_fn
        self.data_aug_fn     = data_aug_fn
        self._mirror_images  = random_mirror
        self.dataset         = dataset
        self.batch_size      = batch_size
        self.image_dir       = image_dir
        self.anno_dir        = anno_dir if anno_dir is not None else image_dir
        self._indexes        = list(range(len(dataset)))
        self.steps_per_epoch = len(self.dataset) // self.batch_size
        self.data = self._load_all_data(image_dir, anno_dir, dataset)
        self.reset(shuffle)

    def reset(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self._indexes)
        else:
            self._indexes = list(range(len(self.dataset)))
        self.current_step = 0

    def _load_all_data(self, image_dir, anno_dir, dataset):
        all_data = []
        pbar = tqdm(dataset)
        pbar.set_description("Loading Data")
        for name in pbar:
            im, an = load_image_anno_pair(image_dir, anno_dir, name)
            pair = {}
            if self.preprocess is not None:
                pair = self.preprocess(im, an)
            else:
                pair['image'] = im
                pair['anno']  = an
            pair["name"] =  name
            all_data.append(pair.copy())
        return all_data

    def get_next_batch(self, augment=True):
        if self.current_step == self.steps_per_epoch:
            print("Data source exhausted, re-init DataGenerator")
            return None, None

        marker = self.current_step * self.batch_size
        batch = self.data_aug_fn(self, marker, self._mirror_images)
        self.current_step += 1
        return self._prepare_batch(batch)

    def _prepare_batch(self, batch):
        images   = batch["images"]
        steering = [ele["steering"] for ele in batch["annotations"]]
        throttle = [ele["throttle"] for ele in batch["annotations"]]
        return images, steering, throttle
