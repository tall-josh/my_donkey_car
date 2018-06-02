import paho.mqtt.client as mqtt
import time
import json

def on_message(client, userdata, msg):
  topic=msg.topic
  payload = str(msg.payload.decode("utf-8", "ignore"))
  commands = json.loads(payload)
  s_command = commands["steering"]
  t_command = commands["throttle"]
  print(f"st: {s_command}, th: {t_command}")

broker = "localhost"
client = mqtt.Client("car_client")
client.on_message=on_message
print(f"Connecting to broker: {broker}")
client.connect(broker)
client.loop_start()
client.subscribe("inference/control")

while True:
  try:
    time.sleep(0.1)
  except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()
    print(f"Fucking off...")
    break
