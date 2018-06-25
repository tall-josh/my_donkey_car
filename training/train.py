from tf_donkey import Model
import os
from utils import *
from generator import DataGenerator, preprocess_normalize_images_bin_annos
from generator import batch_data_aug as data_aug_fn
import json

def update_config(save_dir, payload):
    path = save_dir+"/config.json"
    if os.path.exists(path):
      with open(path, 'r') as f: existing_config=json.load(f)
      payload.update(existing_config)

    with open(path, 'w') as f:
        json.dump(payload, f)

def main():
    import argparse as argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-txt', type=str, required=True)
    parser.add_argument('--test-txt', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--num-bins', type=int, required=True)
    parser.add_argument('--lr', type=float, required=False, default = 0.001)
    parser.add_argument('--batch-size', type=int, required=False, default=50)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--shape', type=int, required=True, nargs=3, help="height width chanels")
    parser.add_argument('--message', type=str, required=True)

    args        = parser.parse_args()
    data_dir    = args.data_dir
    image_dir   = os.path.join(data_dir, "images/")
    anno_dir    = os.path.join(data_dir, "annotations/")
    train_path  = args.train_txt
    test_path   = args.test_txt

    # Load list of image names for train and test
    raw_train   = load_dataset(train_path)
    raw_test    = load_dataset(test_path)

    # Create train and test generators
    num_bins    = args.num_bins
    batch_size  = args.batch_size
    train_gen   = DataGenerator(batch_size=batch_size,
                      dataset=raw_train[:200],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      preprocess_fn=preprocess_normalize_images_bin_annos,
                      data_aug_fn=data_aug_fn,
                      random_mirror=True)
    test_gen    = DataGenerator(batch_size=batch_size,
                      dataset=raw_test[:50],
                      image_dir=image_dir,
                      anno_dir=anno_dir,
                      shuffle=True,
                      preprocess_fn=preprocess_normalize_images_bin_annos,
                      data_aug_fn=data_aug_fn,
                      random_mirror=True)
    # Kick-off
    save_dir    = args.save_dir
    epochs      = args.epochs
    in_shape    = args.shape
    lr          = args.lr
    classes     = [i for i in range(num_bins)]
    message     = args.message

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    update_config(save_dir,
                 {"data_dir"    : data_dir,
                  "num_bins"    : num_bins,
                  "lr"          : lr,
                  "batch_size"  : batch_size,
                  "epochs"      : epochs,
                  "input_shape" : in_shape,
                  "message"     : message})
    car_brain   = Model(in_shape, classes=classes)
    car_brain.train(train_gen, test_gen, save_dir, epochs=epochs)

if __name__ == "__main__":
    main()
