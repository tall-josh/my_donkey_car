# Raspberry Pi3 Setup

The following is a list of steps I used to get Tensorflow 1.8 running from a fresh Raspian 9.0 install.

By default Raspian 9.0 comes with Python2.7 and 3.5. I'm using Python3.5.

The `whl` im using is from [this repo](https://github.com/lhelontra/tensorflow-on-arm/releases)

The final line `...libatlas-base-dev` is thanks to [this thread](https://github.com/Kitt-AI/snowboy/issues/262)

```
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install python3-pip python3-dev git vim -y

wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.8.0/tensorflow-1.8.0-cp35-none-linux_armv7l.whl

sudo pip3 install tensorflow-1.8.0-cp35-none-linux_armv7l.whl

sudo apt-get install libatlas-base-dev -y
```
At this point test your TF install.

```
python3
import tensorflow as tf
x = tf.constant("PLEASE WORK!")

with tf.Session() as sess:
  print(sess.run(x))
```

Now we'll need mosquitto and mqtt for messaging to communicate between scrips/devices.

```
sudo apt-get install mosquitto -y
pip3 install paho-mqtt
```

Opencv

```
sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools libqtgui4 libqtgui4 libqt4-test libwebp-dev libtiff-tools libjasper-dev openexr libgstreamer-plugins-base0.10-0 -y

pip3 install opencv-python
```

Now, if you haven't done so already. Clone this repo:

```
git clone git@github.com:tall-josh/my_donkey_car.git
# or, if https is your flavour
git clone https://github.com/tall-josh/my_donkey_car.git
```

You'll need to update the `BROKER_IP` field in `my_donkey_car/car_control/config.json`
to match the Pi of you Pi. Also, depending on where you've cloned this repo, you may need
to change `FROZEN_GRAPH` and `TENSOR_NAMES` too. In this case the repo has been cloned
into the users home directory.

```
cd my_donkey_car/car_control
vim config.json

# Should look something like:
{"NETWORK_INPUT_SHAPE"      :[1,120,160,3],
	"NUM_STEERING_BINS" : 15,
	"FROZEN_GRAPH"      : "$HOME/my_donkey_car/frozen_graph/frozen.pb",
	"TENSOR_NAMES"      : "$HOME/my_donkey_car/frozen_graph/tensor_names.json",
	"BROKER_IP"         : "<YOUR.IP.ADDRESS>",
        "BROKER_PORT"       :  1883,
        "TOPIC_CONTROL"     : "inference/control",
        "PWM_PERIOD"        : 0.02,
        "STEERING_PIN"      : 1,
        "THROTTLE_PIN"      : 8,
        "STEER_PWM_MIN"     : -0.5,
        "STEER_PWM_MAX"     :  0.5,
        "THROTTLE_PWM_MIN"  : 0.57,
        "THROTTLE_PWM_MAX"  : 0.7}
}
```
