import json
import tensorflow as tf
import os

NUM_BINS = 15

_INPUTS              = "inputs"
_OUTPUTS             = "outputs"
_IMAGE_INPUT         = "image_input"
_STEERING_PREDICTION = "steering_prediction"
_STEERING_PROBS      = "steering_probs"
_THROTTLE_PREDICTION = "throttle_prediction"

def GET_EMPTY_TENSOR_NAME_DICT():
  return {_IMAGE_INPUT         : "",
          _STEERING_PREDICTION : "",
          _STEERING_PROBS      : "",
          _THROTTLE_PREDICTION : ""}

"""
gpu_mask: If on a gpu rig select one or more gpus to use (None to use all)
          ie: gpu_mask="0" for gpu zero or gpu_mask="1,3" for gpus one and three

gpu_fraction: What portion of the gpu memory do you want to use (None to use all)
"""
_GPU_OPTIONS = None
def setup_gpu_options(gpu_mask=None, gpu_fraction=None):
  global _GPU_OPTIONS
  if gpu_mask: os.environ["CUDA_VISIBLE_DEVICES"] = gpu_mask
  if gpu_fraction:
    _GPU_OPTIONS  = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
