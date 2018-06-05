import paho.mqtt.client as mqtt
import time
from load_frozen import load_graph
import tensorflow as tf
import json
import numpy as np
import multiprocessing as mp
import ctypes
from webcam_reader import video_process
from multiprocessing.managers import BaseManager
import cv2
import time

with open("../config.json", 'r') as f:
  CONFIG = json.load(f)

frozen_path = CONFIG["FROZEN_GRAPH"] 
tensor_path = CONFIG["TENSOR_NAMES"]

_graph = load_graph(frozen_path)
_tensors = json.load(open(tensor_path, 'r'))
_image    = _tensors["inputs"]["image_input"]
_steering = _tensors["outputs"]["steering_prediction"]
_throttle = _tensors["outputs"]["throttle_prediction"]

with _graph.as_default() as graph:
  image    = graph.get_tensor_by_name(_image)
  steering = graph.get_tensor_by_name(_steering)
  throttle = graph.get_tensor_by_name(_throttle)
'''
 def on_message(client, userdata, msg):
   topic=msg.topic
   m_decode=str(msg.payload.decode("utf-8","ignore"))
   print(f"message recieved", m_decode)
'''

BROKER        = CONFIG["BROKER_IP"]
PORT          = CONFIG["BROKER_PORT"]
TOPIC_CONTROL = CONFIG["TOPIC_CONTROL"]

print("Connecting to broker: {}:{}".format(BROKER, PORT))
print("Publish topic: '{}'".format(TOPIC_CONTROL))
client = mqtt.Client("video_client")
#client.on_message=on_message
client.connect(BROKER, PORT)
client.loop_start()
print("Connected")

def inference_process(frame_buffer, write_target):
  while True:
    # Toggle write_target, ie: now it's the read target
    write_target.value = not write_target.value
    # read frame from frame_buffer
    frame = frame_buffer[write_target.value]
    # reshape and cast before feeding into the NN
    _frame = np.reshape(frame, CONFIG["NETWORK_INPUT_SHAPE"]).astype(np.float32)
    try:
      # do inference
      with tf.Session(graph=graph) as sess:
        com_steer, com_throt = sess.run([steering, throttle],
                               feed_dict={image: _frame})
      # normalize
      com_steer = float(com_steer/CONFIG["NUM_STEERING_BINS"])
      com_throt = float(com_throt)
      payload = {"steering": com_steer, "throttle": com_throt}
      client.publish(TOPIC_CONTROL, json.dumps(payload))
    except KeyboardInterrupt:
      client.loop_stop()
      client.disconnect()
      print("Fucking off...")
      break


def main():
    flat_image_shape = np.prod(CONFIG["NETWORK_INPUT_SHAPE"])
    frame_0      = mp.Array('f', np.zeros(flat_image_shape, dtype=np.float32), lock=False)
    frame_1      = mp.Array('f', np.zeros(flat_image_shape, dtype=np.float32), lock=False)
    frame_buffer = [frame_0, frame_1]
    write_target = mp.Value(ctypes.c_bool, False)

    p_get_image = mp.Process(target=video_process, args=(frame_buffer, write_target))
    p_get_image.start()
    inference_process(frame_buffer, write_target)
#    p_get_image.join()

if __name__ == "__main__":
    main()
