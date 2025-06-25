#!/usr/bin/env python3
import rospy
import json
import sys
import cv2
import base64
import traceback
import numpy as np
import re
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose, PoseStamped
import tf2_ros
import tf_conversions
import tf2_geometry_msgs
from openai import OpenAI

class ObjectPose:
    def __init__(self, name="", pose=None, width=0, height=0):
        self.name = name
        self.pose = pose if pose else Pose()
        self.width = width
        self.height = height

    def to_dict(self):
        pose_dict = {
            "position": {
                "x": self.pose.position.x,
                "y": self.pose.position.y,
                "z": self.pose.position.z
            },
            "orientation": {
                "x": self.pose.orientation.x,
                "y": self.pose.orientation.y,
                "z": self.pose.orientation.z,
                "w": self.pose.orientation.w
            }
        }
        return {
            "name": self.name,
            "pose": pose_dict,
            "width": self.width,
            "height": self.height
        }

def extract_json_from_gpt_response(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return match.group(1) if match else text

def get_xyz_from_pointcloud(cloud_msg, u, v):
    try:
        gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=[(u, v)])
        point = next(gen, None)
        return point if point and not any(np.isnan(point)) else None
    except Exception as e:
        rospy.logerr(f"PointCloud lookup failed: {e}")
        return None

def get_transformation_from_to(tf_buffer, from_frame: str, to_frame: str):
    try:
        trans = tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0))
        pose_stamped = PoseStamped()
        pose_stamped.header = trans.header
        pose_stamped.pose.position.x = trans.transform.translation.x
        pose_stamped.pose.position.y = trans.transform.translation.y
        pose_stamped.pose.position.z = trans.transform.translation.z
        pose_stamped.pose.orientation = trans.transform.rotation

        quat = trans.transform.rotation
        rot_euler = tf_conversions.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return pose_stamped, rot_euler
    except Exception as e:
        rospy.logerr(f"[TF ERROR] Failed to transform from {from_frame} to {to_frame}: {e}")
        return None, None

def get_enviroment(prompt: str) -> dict:
    rospy.init_node("get_env_node", anonymous=True)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    bridge = CvBridge()

    try:
        rgb_msg = rospy.wait_for_message("/xtion/rgb/image_raw", Image, timeout=5)
        depth_msg = rospy.wait_for_message("/xtion/depth_registered/image_raw", Image, timeout=5)
        pointcloud_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2, timeout=5)

        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        rgb_path = "/tmp/rgb.jpg"
        depth_path = "/tmp/depth.jpg"
        cv2.imwrite(rgb_path, rgb_image)

        depth_norm = np.nan_to_num(depth_image, nan=0.0)
        depth_vis = cv2.normalize(depth_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(depth_path, depth_color)

        with open(rgb_path, "rb") as f1, open(depth_path, "rb") as f2:
            rgb_b64 = base64.b64encode(f1.read()).decode()
            depth_b64 = base64.b64encode(f2.read()).decode()

        client = OpenAI()
        prompt_text = (
            "You are a robot vision system. Given an RGB and a depth image (640x480), describe the environment and identify visible objects.\n"
            "Return JSON with 'description' and 'objects', where each object has 'name' and 'bbox' (pixel coords [x1, y1, x2, y2]).\n"
            "Only output valid JSON. Do not include explanations."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{depth_b64}"}}
                ]
            }]
        )

        gpt_data = json.loads(extract_json_from_gpt_response(response.choices[0].message.content.strip()))
        results = []

        # ✅ 打印当前相机姿态
        camera_pose, camera_euler = get_transformation_from_to(tf_buffer, "xtion_rgb_optical_frame", "base_link")
        if camera_pose:
            rospy.loginfo(f"[Camera] position = ({camera_pose.pose.position.x:.2f}, "
                          f"{camera_pose.pose.position.y:.2f}, {camera_pose.pose.position.z:.2f})")
            rospy.loginfo(f"[Camera] orientation (rpy) = {camera_euler}")

        for obj in gpt_data.get("objects", []):
            name = obj.get("name")
            bbox = obj.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            coords = get_xyz_from_pointcloud(pointcloud_msg, cx, cy)
            if coords:
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = pointcloud_msg.header.frame_id
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.pose.position.x = coords[0]
                pose_msg.pose.position.y = coords[1]
                pose_msg.pose.position.z = coords[2]
                pose_msg.pose.orientation.w = 1.0

                transformed_pose, _ = get_transformation_from_to(tf_buffer, pose_msg.header.frame_id, "torso_lift_link")
                if transformed_pose:
                    obj_pose = ObjectPose(name=name, pose=transformed_pose.pose,
                                          width=x2 - x1, height=y2 - y1)
                    results.append(obj_pose)

        return {
            "description": gpt_data.get("description", ""),
            "objects": results
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Processing failed: {e}"}

def custom_json_serializer(obj):
    if isinstance(obj, ObjectPose):
        return obj.to_dict()
    raise TypeError(f"Type {type(obj)} not serializable")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What do you see?"
    result = get_enviroment(prompt)
    print(json.dumps(result, default=custom_json_serializer, indent=2))

