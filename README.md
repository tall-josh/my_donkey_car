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
