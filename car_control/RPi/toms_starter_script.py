import paho.mqtt.client as mqtt
import time
import json
import numpy as np
from webcam_reader import video_process
from multiprocessing.managers import BaseManager
import cv2
import time

with open("../config.json", 'r') as f:
  CONFIG = json.load(f)

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

def TOMS_CONTROLLER_LOOP():
    """
    Your code goes here
    """


def main():
    TOMS_CONTROLLER_LOOP()

if __name__ == "__main__":
    main()


