import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

# From demo.py
import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

# from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F

# extra imports from visualization.py
import droid_backends
from lietorch import SE3
import geom.projective_ops as pops

bridge = CvBridge()
image_topic = 'image_topic'
counter = 0
lock = False
image = None
intrinsics = []
droid = None

def image_callback(msg, args):
    global lock
    global counter
    global image
    global intrinsics
    global droid

    lock = True

    # Convert the ROS Image message to OpenCV format
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    # Show image
    #cv2.imshow("Image", image)
    #cv2.waitKey(1)

    # Code from image_stream() in demo.py from DROID SLAM repo
    calib = np.loadtxt(args.calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    if len(calib) > 4:
        image = cv2.undistort(image, K, calib[4:])

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1%8, :w1-w1%8]
    image = torch.as_tensor(image).permute(2, 0, 1)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= (w1 / w0)
    intrinsics[1::2] *= (h1 / h0)

    counter += 1
    image = image[None]

    if droid is None:
        args.image_size = [image.shape[2], image.shape[3]]
        droid = Droid(args)

    print("Processing image", counter)
    droid.track(counter, image, intrinsics=intrinsics)
    print("Done processing image", counter)

def timer_callback(event):
    rospy.loginfo("One second has passed!")


def main():
    global lock
    global counter
    global image
    global intrinsics

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()
    args.stereo = False
    # torch.multiprocessing.set_start_method('spawn')

    rospy.init_node('image_array_node', anonymous=True)
    # Subscribe to the "image_raw" topic
    rospy.Subscriber(image_topic, Image, lambda msg: image_callback(msg, args))
    #image_msg = rospy.wait_for_message(image_topic, Image)

    # Timer 
    timer = rospy.Timer(rospy.Duration(1), timer_callback)

    torch.multiprocessing.set_start_method('spawn')

    rate = rospy.Rate(20)  # Set the loop rate in Hz
    while not rospy.is_shutdown():
        # if not lock:
        # image_callback(image_msg, args)
            # lock = False
        # image_callback(image_msg, args)

        rate.sleep()

if __name__ == '__main__':
    main()