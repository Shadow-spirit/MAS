#!/usr/bin/env python3
import rospy
import sys
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs
import sensor_msgs.point_cloud2 as pc2
import geometry_msgs.msg
from moveit_commander import PlanningSceneInterface
from sensor_msgs.msg import PointCloud2
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
import numpy as np

class EnvironmentFromPointCloud:
    def __init__(self):
        rospy.init_node("env_from_pointcloud")

        self.scene = PlanningSceneInterface()
        self.scene_pub = rospy.Publisher("/planning_scene", PlanningScene, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber("/xtion/depth_registered/points", PointCloud2, self.pointcloud_callback)

        rospy.loginfo("EnvironmentFromPointCloud node initialized.")

    def pointcloud_callback(self, cloud_msg):
        try:
            # Step 1: transform到base_link
            transform = self.tf_buffer.lookup_transform("base_link", cloud_msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(cloud_msg, transform)

            # Step 2: 解析点云数据
            points = []
            for p in pc2.read_points(transformed_cloud, field_names=("x", "y", "z"), skip_nans=True):
                if not np.isfinite(p[2]):
                    continue
                points.append([p[0], p[1], p[2]])

            if len(points) < 10:
                rospy.logwarn("PointCloud too small, skipping.")
                return

            points = np.array(points)

            min_xyz = np.min(points, axis=0)
            max_xyz = np.max(points, axis=0)

            center = (min_xyz + max_xyz) / 2.0
            size = max_xyz - min_xyz

            rospy.loginfo(f"Environment bounds: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")

            # Step 3: 清除旧的障碍物
            self.scene.remove_world_object()
            rospy.sleep(1.0)

            # Step 4: 创建新的障碍盒子
            co = CollisionObject()
            co.header.frame_id = "base_link"
            co.id = "environment_box"

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [size[0], size[1], size[2]]

            pose = geometry_msgs.msg.Pose()
            pose.position.x = center[0]
            pose.position.y = center[1]
            pose.position.z = center[2]
            pose.orientation.w = 1.0

            co.primitives.append(primitive)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD

            planning_scene = PlanningScene()
            planning_scene.world.collision_objects.append(co)
            planning_scene.is_diff = True

            self.scene_pub.publish(planning_scene)

            rospy.loginfo("Planning Scene updated with environment box.")

        except Exception as e:
            rospy.logerr(f"Failed processing pointcloud: {e}")

if __name__ == "__main__":
    try:
        env_updater = EnvironmentFromPointCloud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

