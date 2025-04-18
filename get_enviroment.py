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
from sensor_msgs.msg import Image, CameraInfo
from openai import OpenAI
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped  # 使用官方的 geometry_msgs

# 自定义 ObjectPose 类
class ObjectPose:
    def __init__(self, name="", pose=None, width=0, height=0):
        self.name = name
        self.pose = pose if pose else Pose()  # 默认pose
        self.width = width
        self.height = height

    def to_dict(self):
        # 将 Pose 对象转换为字典
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

# 自定义 ObjectPoseList 类
class ObjectPoseList:
    def __init__(self):
        self.objects = []
        self.header = None

    def to_dict(self):
        return {"header": self.header, "objects": [obj.to_dict() for obj in self.objects]}

# 解析 GPT 返回的 JSON 格式
def extract_json_from_gpt_response(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return match.group(1) if match else text

def get_camera_intrinsics(timeout=5):
    cam_info = rospy.wait_for_message("/xtion/rgb/camera_info", CameraInfo, timeout=timeout)
    K = cam_info.K
    return {"fx": K[0], "fy": K[4], "cx": K[2], "cy": K[5]}

def get_xyz_from_pixel(u, v, depth_image, intrinsics):
    depth = depth_image[v, u]
    if np.isnan(depth) or np.isinf(depth) or depth == 0:
        return None
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return [float(x), float(y), float(z)]

def transform_pose(pose_stamped, target_frame, tf_buffer):
    rospy.loginfo(f"Transforming pose to frame: {target_frame}")
    try:
        transform = tf_buffer.lookup_transform(target_frame,
                                                pose_stamped.header.frame_id,
                                                rospy.Time(0),
                                                rospy.Duration(1.0))
        transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        rospy.loginfo("Pose transformed successfully")
        return transformed_pose
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"Transform failed: {e}")
        return None

def get_enviroment(prompt: str) -> dict:
    rospy.init_node("get_env_node", anonymous=True)

    # Initialize tf_buffer and tf_listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    bridge = CvBridge()

    try:
        # Step 1: 获取图像
        rgb_msg = rospy.wait_for_message("/xtion/rgb/image_raw", Image, timeout=5)
        depth_msg = rospy.wait_for_message("/xtion/depth_registered/image_raw", Image, timeout=5)

        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        # Step 2: 保存图像
        rgb_path = "/tmp/rgb.jpg"
        depth_path = "/tmp/depth.jpg"
        cv2.imwrite(rgb_path, rgb_image)

        depth_clean = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(depth_path, depth_color)

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Failed to capture ROS images: {e}"}

    try:
        with open(rgb_path, "rb") as f1, open(depth_path, "rb") as f2:
            rgb_b64 = base64.b64encode(f1.read()).decode()
            depth_b64 = base64.b64encode(f2.read()).decode()
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Failed to encode images: {e}"}

    try:
        client = OpenAI()
        prompt_text = (
            "You are a robot vision system. Given an RGB and a depth image, describe the environment and identify visible objects.If you cannot see specific object, you could recommend a direction to movet the robot head in description part(up,low,right,left)\n"
            "Return JSON format like:\n"
            "{\n"
            "  \"description\": \"...\",\n"
            "  \"objects\": [\n"
            "    {\"name\": \"red cube\", \"position_in_image\": [320, 240]}\n"
            "  ]\n"
            "}\n"
            "Reply ONLY with JSON. Do not explain or format."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{depth_b64}"}}
                    ]
                }
            ],
            max_tokens=700,
        )

        raw_content = response.choices[0].message.content
        json_text = extract_json_from_gpt_response(raw_content.strip())
        gpt_data = json.loads(json_text)

    except Exception as e:
        traceback.print_exc()
        return {"error": f"GPT processing failed: {e}"}

    try:
        intrinsics = get_camera_intrinsics()
        description = gpt_data.get("description", "")
        results = []

        for obj in gpt_data.get("objects", []):
            name = obj.get("name")
            u, v = obj.get("position_in_image", [0, 0])
            coords = get_xyz_from_pixel(int(u), int(v), depth_image, intrinsics)
            if coords:
                pose_msg = PoseStamped()
                pose_msg.header = rgb_msg.header
                pose_msg.pose.position.x = coords[0]
                pose_msg.pose.position.y = coords[1]
                pose_msg.pose.position.z = coords[2]
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0

                # Transform to base_footprint frame
                transformed_pose = transform_pose(pose_msg, "base_footprint", tf_buffer)
                if transformed_pose:
                    object_pose = ObjectPose()
                    object_pose.name = name
                    object_pose.pose = transformed_pose.pose
                    results.append(object_pose)

        object_pose_list = ObjectPoseList()
        object_pose_list.objects = results
        object_pose_list.header = rgb_msg.header

        return {"description": description, "objects": results}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"3D position computation failed: {e}"}


def custom_json_serializer(obj):
    if isinstance(obj, ObjectPose):
        return obj.to_dict()
    if isinstance(obj, ObjectPoseList):
        return obj.to_dict()
    raise TypeError(f"Type {type(obj)} not serializable")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What do you see?"
    result = get_enviroment(prompt)
    print(json.dumps(result, default=custom_json_serializer))


