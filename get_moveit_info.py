#!/usr/bin/python3.8

import rospy
import moveit_commander
import sys
import json

def get_moveit_info():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_info", anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("arm_torso")

    info = {
        "planning_frame": group.get_planning_frame(),
        "end_effector_link": group.get_end_effector_link(),
        "group_names": robot.get_group_names(),
        "current_pose": {
            "position": {
                "x": group.get_current_pose().pose.position.x,
                "y": group.get_current_pose().pose.position.y,
                "z": group.get_current_pose().pose.position.z
            },
            "orientation": {
                "x": group.get_current_pose().pose.orientation.x,
                "y": group.get_current_pose().pose.orientation.y,
                "z": group.get_current_pose().pose.orientation.z,
                "w": group.get_current_pose().pose.orientation.w
            }
        }
    }

    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    get_moveit_info()

