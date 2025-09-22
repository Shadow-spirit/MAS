#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time
import rospy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np

class ArucoPixelPublisher:
    def __init__(self, target_marker_id, image_topic, aruco_dict_name, oneshot=False):
        self.bridge = CvBridge()
        self.target_marker_id = target_marker_id
        self.image_topic = image_topic
        self.oneshot = oneshot
        self.found_once = False
        self.last_frame_time = time.time()
        self.frame_count = 0

        # ---- ArUco init: support new (4.7+) and legacy API ----
        self.dict_id = getattr(aruco, aruco_dict_name)
        self.aruco_dict = aruco.getPredefinedDictionary(self.dict_id)
        try:
            self.params = aruco.DetectorParameters_create()
        except AttributeError:
            self.params = aruco.DetectorParameters()

        self.detector = None
        if hasattr(aruco, "ArucoDetector"):
            # New API in OpenCV 4.7+ (and 4.11). This replaces aruco.detectMarkers
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.params)

        # ---- Publisher ----
        self.pub = rospy.Publisher('/gripper_cam_aruco_pose', PointStamped, queue_size=1)

        # ---- Subscriber: auto-select raw or compressed ----
        if image_topic.endswith("/compressed"):
            self.sub = rospy.Subscriber(image_topic, CompressedImage, self.image_callback_compressed, queue_size=1)
        else:
            self.sub = rospy.Subscriber(image_topic, Image, self.image_callback_raw, queue_size=1)

        rospy.loginfo("Looking for ArUco marker ID %s on %s with dict %s (OpenCV %s) ...",
                      target_marker_id, image_topic, aruco_dict_name, cv2.__version__)

    def _detect(self, gray):
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        elif hasattr(aruco, "detectMarkers"):
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        else:
            raise RuntimeError("This OpenCV build lacks ArUco detectMarkers and ArucoDetector. Install opencv-contrib.")
        return corners, ids

    def _process_frame(self, cv_image):
        self.frame_count += 1
        self.last_frame_time = time.time()

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        try:
            corners, ids = self._detect(gray)
        except Exception as e:
            rospy.logwarn("ArUco detect error: %s", e)
            return

        if ids is None or len(ids) == 0:
            return
        ids = np.array(ids).flatten().tolist()

        if self.target_marker_id in ids:
            idx = ids.index(self.target_marker_id)
            mc = corners[idx][0]   # (4,2)
            avg_x = float(np.mean(mc[:, 0]))
            avg_y = float(np.mean(mc[:, 1]))

            pose_msg = PointStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "camera_frame"
            pose_msg.point.x = avg_x
            pose_msg.point.y = avg_y
            pose_msg.point.z = 0.0
            self.pub.publish(pose_msg)

            if not self.found_once:
                rospy.loginfo("Found marker %s at (u=%.1f, v=%.1f) → /gripper_cam_aruco_pose",
                              self.target_marker_id, avg_x, avg_y)
                self.found_once = True

            if self.oneshot:
                rospy.signal_shutdown("One-shot publish done.")

    def image_callback_raw(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge convert error (raw): %s", e)
            return
        self._process_frame(cv_image)

    def image_callback_compressed(self, msg: CompressedImage):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge convert error (compressed): %s", e)
            return
        self._process_frame(cv_image)

    def spin_with_watchdog(self):
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            if time.time() - self.last_frame_time > 2.0 and self.frame_count == 0:
                rospy.logwarn("No frames received from %s. Check the camera topic name.", self.image_topic)
            rate.sleep()

def main():
    parser = argparse.ArgumentParser(description="Gripper-cam ArUco pixel publisher (OpenCV 4.7+ compatible)")
    parser.add_argument("marker_id", type=int, help="Target ArUco ID to track")
    parser.add_argument("--image-topic", default="/camera/image_raw", help="Camera image topic (raw or .../compressed)")
    parser.add_argument("--dict", default="DICT_4X4_100", help="OpenCV aruco dictionary (e.g., DICT_4_4_50, DICT_4X4_100, DICT_5X5_100, etc.)")
    parser.add_argument("--oneshot", action="store_true", help="Publish once then shutdown")
    args = parser.parse_args()

    rospy.init_node('gripper_cam_detector', anonymous=True)
    node = ArucoPixelPublisher(
        target_marker_id=args.marker_id,
        image_topic=args.image_topic,
        aruco_dict_name=args.dict,
        oneshot=args.oneshot
    )
    node.spin_with_watchdog()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
