import cv2
import multiprocessing as mp
import ctypes
import numpy as np
import json
import time

NORM = 1. / 255.

with open("../config.json", 'r') as f:
    CONFIG = json.load(f)

def resize(img, out_size):
    shape = np.shape(img)
    orig_h = shape[0]
    h_scale = out_size[0] / orig_h

    orig_w = shape[1]
    w_scale = out_size[1] / orig_w
  
    return cv2.resize(img, None, fx=w_scale, fy=h_scale)

def video_process(frame_buffer, write_target):
    cam = cv2.VideoCapture(0)
    while True:
        try:
            ret_val, img = cam.read()
            if ret_val:
                img = resize(img, CONFIG["NETWORK_INPUT_SHAPE"][1:])
                img = img.flatten()[:] * NORM
                wt = int(write_target.value)
                frame_buffer[wt][:] = img
              
                print("FRAME: {}, [{}]".format(frame_buffer[wt][60*80], wt))
            else:
                print("false")
        except KeyboardInterrupt:
            print("Video read, fucking off...")
            break

def main():
    flat_image_shape = np.prod(CONFIG["NETWORK_INPUT_SHAPE"])
    frame = mp.Array(ctypes.c_float, list(np.zeros(flat_image_shape)), lock=False)
    p     = mp.Process(target=video_process, args=(frame,))
    p.start()

    while True:
        try:
            middle_pix = 28800
            #print("Frame: {}".format(frame[middle_pix]))
        except KeyboardInterrupt:
            print("Video read MAIN, fucking off...")
            break


if __name__ == "__main__":
    main()
