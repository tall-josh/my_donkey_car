import paho.mqtt.client as mqtt
import time
from load_frozen import load_graph
import tensorflow as tf
import json
import numpy as np

frozen_path = "/home/jp/Documents/FYP/ml/car_control/donkey_brain/frozen.pb"
tensor_path = "/home/jp/Documents/FYP/ml/car_control/donkey_brain/tensor_names.json"
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
broker = "localhost"
client = mqtt.Client("video_client")
#client.on_message=on_message
print(f"Connecting to broker: {broker}")
client.connect(broker)
client.loop_start()

while True:
  DUMMY_IMAGE = np.random.rand(1,120,160,3)
  try:
    with tf.Session(graph=graph) as sess:
      com_steer, com_throt = sess.run([steering, throttle],
                               feed_dict={image: DUMMY_IMAGE})
    # normalize
    com_steer = float(com_steer/14.)
    com_throt = float(com_throt)
    payload = {"steering": com_steer, "throttle": com_throt}
    client.publish("inference/control", json.dumps(payload))
  except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()
    print(f"Fucking off...")
    break
