#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_marker_tools.py
ArUco-based pick/place helpers for TIAGo (ROS1, Python3, OpenCV).
- Locate a marker in 3D (requires depth + camera_info + tf2)
- Pick up by marker id
- Place by marker id

Dependencies (Ubuntu/ROS Noetic typical):
  sudo apt-get install ros-noetic-cv-bridge ros-noetic-image-transport \
                       ros-noetic-image-geometry ros-noetic-tf ros-noetic-tf2-ros
  pip3 install opencv-python
"""

import os
import time
import numpy as np
import rospy
import tf2_ros
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import subprocess

# ----------------------- Defaults (override via ROS params) -----------------------
DEFAULTS = {
    "head": {
        "rgb_topic": "/xtion/rgb/image_raw",
        "depth_topic": "/xtion/depth_registered/image_raw",
        "camera_info_topic": "/xtion/rgb/camera_info",
        "optical_frame": "xtion_rgb_optical_frame",
    },
    "gripper": {
        "rgb_topic": "/gripper_camera/image_raw",
        "depth_topic": "/gripper_camera/depth/image_raw",
        "camera_info_topic": "/gripper_camera/camera_info",
        "optical_frame": "gripper_rgb_optical_frame",
    },
    "base_frame": "base_link",
}

def _param(ns, key, default):
    return rospy.get_param(f"{ns}/{key}", default)

def _get_cam_cfg(which):
    cfg = DEFAULTS[which].copy()
    base = f"/aruco/{which}"
    for k, v in cfg.items():
        cfg[k] = _param(base, k, v)
    cfg["base_frame"] = _param("/aruco", "base_frame", DEFAULTS["base_frame"])
    return cfg

# ----------------------- Core helpers -----------------------
_bridge = CvBridge()

def _get_camera_matrix(cam_info):
    K = np.array(cam_info.K, dtype=np.float32).reshape(3, 3)
    return K

def _depth_to_3d(K, depth_m, u, v):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float32)

def _transform_point(tf_buffer, xyz_cam, from_frame, to_frame):
    stamped = PointStamped()
    stamped.header.stamp = rospy.Time(0)
    stamped.header.frame_id = from_frame
    stamped.point.x, stamped.point.y, stamped.point.z = xyz_cam.tolist()
    trans = tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0))
    # Apply transform
    import tf2_py as tf2
    m = tf2.TransformerCore()
    from geometry_msgs.msg import TransformStamped
    T = TransformStamped()
    T.header = trans.header
    T.child_frame_id = trans.child_frame_id
    T.transform = trans.transform
    m.setTransform(T)
    out = m.transformPoint(to_frame, stamped)
    return np.array([out.point.x, out.point.y, out.point.z], dtype=np.float32)

def _detect_aruco(image_bgr, marker_id, dictionary=cv2.aruco.DICT_4X4_50):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None or len(ids) == 0:
        return None, None
    ids = ids.flatten()
    for i, mid in enumerate(ids):
        if int(mid) == int(marker_id):
            pts = corners[i].reshape(4, 2)
            c = pts.mean(axis=0)
            return c, pts
    return None, None

# ----------------------- Public API -----------------------
def locate_marker_3d(marker_id, camera="head"):
    """
    Returns dict with 3D position of the ArUco marker in base frame:
      { "x": ..., "y": ..., "z": ..., "u": px, "v": px, "frame": "base_link" }
    """
    try:
        rospy.init_node("aruco_localizer", anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException:
        pass

    cfg = _get_cam_cfg(camera)
    rgb = rospy.wait_for_message(cfg["rgb_topic"], Image, timeout=5.0)
    depth = rospy.wait_for_message(cfg["depth_topic"], Image, timeout=5.0)
    info = rospy.wait_for_message(cfg["camera_info_topic"], CameraInfo, timeout=5.0)

    img = _bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
    depth_img = _bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
    K = _get_camera_matrix(info)

    center, _ = _detect_aruco(img, marker_id)
    if center is None:
        return {"error": f"marker {marker_id} not found"}

    u, v = int(center[0]), int(center[1])
    d = float(depth_img[v, u])
    if d > 10.0:  # mm
        d /= 1000.0
    if not np.isfinite(d) or d <= 0.0:
        return {"error": f"invalid depth at ({u},{v})"}

    xyz_cam = _depth_to_3d(K, d, u, v)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.2)
    xyz_base = _transform_point(tf_buffer, xyz_cam, cfg["optical_frame"], cfg["base_frame"])
    return {"x": float(xyz_base[0]), "y": float(xyz_base[1]), "z": float(xyz_base[2]), "u": u, "v": v, "frame": cfg["base_frame"]}

def _move_arm_to_pose(x, y, z, qx=0, qy=0, qz=0, qw=0, timeout=20):
    here = os.path.dirname(__file__)
    tool = os.path.join(here, "move_arm_tool.py")
    cmd = [tool, str(x), str(y), str(z), str(qx), str(qy), str(qz), str(qw)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        return ("SUCCESS" in out)
    except Exception as e:
        rospy.logerr(f"move_arm_tool error: {e}")
        return False

def _control_pal_gripper(left=0.04, right=0.04):
    try:
        rospy.init_node('pal_gripper_controller_from_aruco', anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException:
        pass
    pub = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=10)
    rospy.sleep(0.5)
    traj = JointTrajectory()
    traj.joint_names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
    pt = JointTrajectoryPoint()
    pt.positions = [left, right]
    pt.time_from_start = rospy.Duration(1.0)
    traj.points.append(pt)
    pub.publish(traj)
    return True

def pickup_by_marker(marker_id, camera="head", approach=0.12):
    loc = locate_marker_3d(marker_id, camera=camera)
    if "error" in loc:
        return f"pickup_by_marker failed: {loc['error']}"
    x, y, z = loc["x"], loc["y"], loc["z"]
    _control_pal_gripper(0.04, 0.04)  # open
    if not _move_arm_to_pose(x, y, z + approach):
        return "pickup_by_marker: failed to move above marker."
    if not _move_arm_to_pose(x, y, max(0.2, z + 0.02)):
        return "pickup_by_marker: failed to descend."
    _control_pal_gripper(0.0, 0.0)  # close
    time.sleep(0.5)
    _move_arm_to_pose(x, y, z + approach)
    return f"Picked up marker {marker_id} at ({x:.3f},{y:.3f},{z:.3f})"

def place_by_marker(marker_id, camera="head", approach=0.12, open_width=0.04):
    loc = locate_marker_3d(marker_id, camera=camera)
    if "error" in loc:
        return f"place_by_marker failed: {loc['error']}"
    x, y, z = loc["x"], loc["y"], loc["z"]
    if not _move_arm_to_pose(x, y, z + approach):
        return "place_by_marker: failed to move above target."
    if not _move_arm_to_pose(x, y, max(0.2, z + 0.02)):
        return "place_by_marker: failed to descend."
    _control_pal_gripper(open_width, open_width)  # release
    time.sleep(0.5)
    _move_arm_to_pose(x, y, z + approach)
    return f"Placed object at marker {marker_id} at ({x:.3f},{y:.3f},{z:.3f})"

