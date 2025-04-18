#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import sys

def get_latest_rgbd_image(save_path="/tmp/rgbd.jpg", timeout=5):
    rospy.init_node('get_rgbd_image_node', anonymous=True)
    bridge = CvBridge()
    try:
        rgb_msg = rospy.wait_for_message("/xtion/rgb/image_raw", Image, timeout=timeout)
        depth_msg = rospy.wait_for_message("/xtion/depth_registered/image_raw", Image, timeout=timeout)

        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        # 替换非法值，防止后续处理报错
        depth_clean = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)

        # 归一化到 0-255 范围用于可视化
        depth_vis = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        # Combine images side by side
        rgbd = np.concatenate((rgb_image, depth_vis_color), axis=1)
        cv2.imwrite(save_path, rgbd)
        sys.stdout.write(json.dumps({"image_path": save_path}))
    except Exception as e:
        sys.stdout.write(json.dumps({"error": f"Failed to get RGBD image: {e}"}))

if __name__ == "__main__":
    get_latest_rgbd_image()

