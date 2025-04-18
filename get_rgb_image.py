#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import json
def get_latest_rgb_image(save_path="/tmp/rgb.jpg", timeout=5):
    rospy.init_node('get_rgb_image_node', anonymous=True)
    bridge = CvBridge()
    try:
        msg = rospy.wait_for_message("/xtion/rgb/image_raw", Image, timeout=timeout)
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(save_path, cv_img)
        print(json.dumps({"image_path": "/tmp/rgb.jpg"}))
    except Exception as e:
        print(f"[ERROR] Failed to get image: {e}")

if __name__ == "__main__":
    get_latest_rgb_image()

