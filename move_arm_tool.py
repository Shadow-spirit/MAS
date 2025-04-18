#!/usr/bin/python3.8
import rospy
import moveit_commander
import geometry_msgs.msg
import sys
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
import numpy as np
def move_arm(x, y, z, qx=0, qy=0, qz=0, qw=0):
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_arm_node", anonymous=True)

    group = moveit_commander.MoveGroupCommander("arm_torso")
    group.set_pose_reference_frame("base_link")

    pose = geometry_msgs.msg.Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)

    if float(qx) == float(qy) == float(qz) == float(qw) == 0.0:
    # use default hand-down orientation
        pose.orientation = geometry_msgs.msg.Quaternion(*quaternion_from_euler(0, np.pi/2, 0))
    else:
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
 
    group.set_pose_target(pose)
    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    print("SUCCESS" if success else "FAILURE")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Error: Usage: move_arm_ros.py x y z qx qy qz qw")
        sys.exit(1)

    x, y, z, qx, qy, qz, qw = sys.argv[1:8]
    move_arm(x, y, z, qx, qy, qz, qw)

