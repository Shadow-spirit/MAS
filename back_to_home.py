#! /usr/bin/env python
import rospy
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def t_create_action_goal(trajectory):
    goal = FollowJointTrajectoryActionGoal()
    goal.goal.trajectory = trajectory
    goal.header.stamp = rospy.Time.now()
    return goal

def t_create_trajectory(joint_names, positions_list, times_from_start):
    trajectory = JointTrajectory()
    trajectory.joint_names = joint_names
    
    for positions, time in zip(positions_list, times_from_start):
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(time)
        trajectory.points.append(point)
    
    return trajectory

def t_publish_trajectories(torso_pub,arm_pub):

    torso_trajectory = t_create_trajectory(
        ["torso_lift_joint"],
        [[0.35]],
        [10.0]
    )
    
    arm_trajectory = t_create_trajectory(
        ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"],
        [
            [0.1, 0.4, -1.41, 1.71, 0.43, -1.37, 1.7]
        ],
        [10.0]
    )
    
    torso_goal = t_create_action_goal(torso_trajectory)
    arm_goal = t_create_action_goal(arm_trajectory)
    
    torso_pub.publish(torso_goal)
    arm_pub.publish(arm_goal)
    rospy.loginfo("Trajectories published")

def main():
    rospy.init_node('trajectory_pub', anonymous=True)
    arm_pub = rospy.Publisher('/arm_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
    torso_pub = rospy.Publisher('/torso_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
        # 等待发布器就绪
    rospy.sleep(1)
    
    # 发布一次完整的轨迹
    t_publish_trajectories(torso_pub,arm_pub)
    
    rospy.loginfo("Trajectory execution completed. Press Ctrl+C to exit.")
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
