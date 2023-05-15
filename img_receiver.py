import socket
import cv2
import numpy as np
import threading
import queue

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

torch.multiprocessing.set_start_method('spawn')





# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
server_address = ('localhost', 8000) # replace with your desired IP address and port number
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)
connection, address = sock.accept()

# Create a window to display the images
cv2.namedWindow("Screenshot", cv2.WINDOW_NORMAL)

frame_queue = queue.Queue()

droid = None
counter = 0

def process_image(image):
    global droid
    global counter

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

def process_frames():
    while True:
        # Get the next frame from the queue
        image = frame_queue.get()
        
        # Wait until the previous frame has finished processing
        if not process_frames.locked():
            process_frames.locked().acquire()
        else:
            continue
        
        try:
            # Process the frame using the 3D SLAM model
            # 3D SLAM code here

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
            
            # Release the lock to allow the next frame to be processed
            process_frames.locked().release()
        except:
            # Release the lock in case of an exception
            process_frames.locked().release()


# Initialize the lock
process_frames.locked = threading.Lock

# Start the thread to process the frame
processing_thread = threading.Thread(target=process_frames)
processing_thread.start()


while True:
    connection, client_address = sock.accept()

    try:
        # Receive the screenshot bytes from the client
        screenshotBytes = b""
        while True:
            data = connection.recv(1024)
            if not data:
                break
            screenshotBytes += data
        
            
        # Convert the bytes to an image
        screenshotArray = np.frombuffer(screenshotBytes, dtype=np.uint8)
        screenshotImage = cv2.imdecode(screenshotArray, cv2.IMREAD_COLOR)
        
        # Add the image to the queue
        # frame_queue.put(screenshotImage)


        # Display the image
        cv2.imshow("Screenshot", screenshotImage)
        cv2.waitKey(1)

        process_image(screenshotImage)

    finally:
        # Clean up the connection
        connection.close()
