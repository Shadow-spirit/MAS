#!/usr/bin/env python3.10

import asyncio
import rospy
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import subprocess
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import BaseChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import json
import yaml

# === Load available object/location names from object_aruco_low.yaml and prepend to coder system prompt ===
try:
    _objmap_paths = ["object_aruco_low.yaml", "./config/object_aruco_low.yaml", "/mnt/data/object_aruco_low.yaml"]
    _objmap = None
    for _p in _objmap_paths:
        if os.path.exists(_p):
            with open(_p, "r") as _f:
                _objmap = yaml.safe_load(_f) or {}
            break
    _names = ", ".join(sorted(_objmap.keys())) if _objmap else "NONE"
    coder_system_message_prefix = f"Allowed names: [{_names}].\n"
except Exception as _e:
    coder_system_message_prefix = "Allowed names: [config not found].\n"

from simple_aruco_executor import pick_and_place_by_name, locate_by_name, list_available_objects
import base64
from openai import OpenAI
from typing import Dict
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2,Image
from control_msgs.msg import FollowJointTrajectoryActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import re
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.ui import Console

from memory_tool import (
    store_triples, store_image_link, query_graph, list_entities,
    search_visual_memory_from_image, get_entity_by_vector_id, describe_image_from_path
)

def safe_init_node(name):
    if not rospy.core.is_initialized():
        rospy.set_param('/use_sim_time', False)
        rospy.init_node(name, anonymous=True, disable_signals=True)

def grasp_item(x: float, y: float, z: float) -> str:
    """
    Performs a grasp action by first hovering above the target, then descending and closing the gripper.
    """
    # Hover first
    hover_result = move_arm_to_pose(x, y, z + 0.1)
    if "failed" in hover_result.lower():
        hover_result_retry = move_arm_to_pose(x, y, z + 0.1)
        if "failed" in hover_result_retry.lower():
            return "Tried to hover above object, failed twice. to commander"

    # Descend
    descend_result = move_arm_to_pose(x, y, z)
    if "failed" in descend_result.lower():
        descend_result_retry = move_arm_to_pose(x, y, z)
        if "failed" in descend_result_retry.lower():
            return "Tried to descend for grasping, failed twice. to commander"

    # Close gripper
    grip_result = control_pal_gripper(joint_positions={"gripper_left_finger_joint": 0.0, "gripper_right_finger_joint": 0.0})
    return f"Grasp action completed. Hover → descend → close. Gripper result: {grip_result}"


def move_head(pan: float, tilt: float) -> str:
    """
    Move the TIAGo's head to a specified pan and tilt angle.

    Args:
        pan (float): The desired angle (in radians) for head_1_joint (left-right).
        tilt (float): The desired angle (in radians) for head_2_joint (up-down).

    Returns:
        str: Success or failure message.
    """
    import rospy
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    try:
        rospy.init_node('move_head_node', anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException:
        pass  # Already initialized

    pub = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=10)
    rospy.sleep(1.0)

    traj = JointTrajectory()
    traj.joint_names = ['head_1_joint', 'head_2_joint']

    point = JointTrajectoryPoint()
    point.positions = [pan, tilt]
    point.time_from_start = rospy.Duration(1.0)
    traj.points.append(point)

    try:
        pub.publish(traj)
        return f"Head moved to pan={pan}, tilt={tilt}"
    except Exception as e:
        return f"Failed to move head: {e}"


def reset_entire_arm_posture() -> str:
    import rospy
    from control_msgs.msg import FollowJointTrajectoryActionGoal
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    def create_goal(joint_names, positions, time_from_start):
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(time_from_start)
        trajectory.points.append(point)

        goal = FollowJointTrajectoryActionGoal()
        goal.goal.trajectory = trajectory
        goal.header.stamp = rospy.Time.now()
        return goal

    try:
        rospy.init_node('reset_arm_posture_node', anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException:
        pass  # already initialized

    torso_pub = rospy.Publisher('/torso_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
    arm_pub = rospy.Publisher('/arm_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
    
    rospy.sleep(1.0)

    # 设置目标位置（默认姿态）
    torso_goal = create_goal(
        ["torso_lift_joint"],
        [0.1],  # 设置腰部高度
        5.0
    )

    arm_goal = create_goal(
        ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"],
        [0.1, 0.4, -1.41, 1.71, 0.43, -1.37, 1.7],  # 修改角度以使肘部朝上
        10.0
    )

    torso_pub.publish(torso_goal)
    arm_pub.publish(arm_goal)

    return "Default posture with elbow up command sent to TIAGo arm and torso."



def get_moveit_status() -> dict:
    """
    Returns current MoveIt status: pose, joints, reference frame, and end-effector link.
    """
    import rospy
    import moveit_commander

    try:
        rospy.init_node("get_moveit_status_node", anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException:
        pass  # Node already initialized

    moveit_commander.roscpp_initialize([])

    group = moveit_commander.MoveGroupCommander("arm_torso")

    status = {
        "reference_frame": group.get_pose_reference_frame(),
        "end_effector_link": group.get_end_effector_link(),
        "current_joint_values": group.get_current_joint_values(),
        "current_pose": {
            "position": {
                "x": group.get_current_pose().pose.position.x,
                "y": group.get_current_pose().pose.position.y,
                "z": group.get_current_pose().pose.position.z,
            },
            "orientation": {
                "x": group.get_current_pose().pose.orientation.x,
                "y": group.get_current_pose().pose.orientation.y,
                "z": group.get_current_pose().pose.orientation.z,
                "w": group.get_current_pose().pose.orientation.w,
            },
        }
    }

    return status



def control_pal_gripper(joint_positions: Dict[str, float]) -> str:
    """
    Publishes a joint trajectory command to control the PAL Gripper on TIAGo using ROS.

    Args:
        joint_positions (dict): A dictionary of joint names and their target positions.

    Returns:
        str: A message confirming the command or indicating an error.
    """
    import rospy
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    # Allowed joints for PAL Gripper on TIAGo
    allowed_joints = ["gripper_left_finger_joint", "gripper_right_finger_joint"]

    try:
        rospy.init_node('pal_gripper_controller', anonymous=True, disable_signals=True)
    except rospy.exceptions.ROSException:
        pass  # Node already initialized

    pub = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=10)
    rospy.sleep(1.0)  # Allow publisher connection

    # Filter joints
    filtered_joints = [j for j in allowed_joints if j in joint_positions]
    if not filtered_joints:
        return f"No valid joints provided. Allowed joints: {allowed_joints}"

    traj = JointTrajectory()
    traj.joint_names = filtered_joints

    point = JointTrajectoryPoint()
    point.positions = [joint_positions[j] for j in filtered_joints]
    point.time_from_start = rospy.Duration(1.0)

    traj.points.append(point)

    try:
        pub.publish(traj)
        return f"Command sent to PAL gripper: {dict(zip(filtered_joints, point.positions))}"
    except Exception as e:
        return f"Failed to send command: {e}"






def extract_json_from_gpt_response(text: str) -> str:
    # 从返回的文本中提取 JSON 部分
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    return match.group(1) if match else text

def get_enviroment(prompt: str) -> dict:
    try:
        # 通过 subprocess 调用 get_enviroment 脚本
        result = subprocess.check_output(
            ['python3', 'get_enviroment.py', prompt],
            text=True,
            stderr=subprocess.STDOUT,  # 捕获错误输出
            timeout=20
        )
        print("📤 Subprocess 返回:", repr(result))  # 调试输出

        # 清理并提取 JSON 数据
        json_text = extract_json_from_gpt_response(result.strip())
        gpt_data = json.loads(json_text)
        
        return gpt_data

    except subprocess.CalledProcessError as e:
        print("❌ Subprocess failed:", e.output)
        return {"error": f"Script failed: {e.output}"}

    except json.JSONDecodeError as e:
        print("❌ JSON decode failed:", e)
        return {"error": f"Invalid JSON output: {e}"}

    except Exception as e:
        print("❌ Unexpected error:", e)
        return {"error": f"Unhandled error: {e}"}

def move_arm_to_pose(x: float,y: float,z: float,qx: float = 0.0,qy: float = 0.0,qz: float = 0.0,qw: float = 0.0) -> str:
    script_path = os.path.join(os.path.dirname(__file__), "move_arm_tool.py")
    cmd = [script_path, str(x), str(y), str(z), str(qx), str(qy), str(qz), str(qw)]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=20, text=True)
        return "Arm moved successfully." if "SUCCESS" in output else "Arm movement failed."
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output}"
    except FileNotFoundError:
        return f"Error: move_arm_tool.py not found at path: {script_path}"


def run_rostopic_echo(topic: str) -> str:
    print(f"[TOOL] run_rostopic_echo called on {topic}")
    try:
        output = subprocess.check_output(["rostopic", "echo", "-n1", topic], timeout=5.0)
        return output.decode("utf-8")
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"
    except subprocess.TimeoutExpired:
        return "Timeout: No message received on topic."

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    user = UserProxyAgent(name="user")

    commander = AssistantAgent(
        name="commander",
        model_client=model_client,
        handoffs=["user", "coder", "memory"],
        tools=[],
        system_message="""
You are the Commander in a multi-agent TIAGo robot system.

Responsibilities:
- Interpret the user's high-level instructions (e.g., "grab the cup", "wave", "look around").
- Pass the core intent to the Coder agent without requiring specific parameters.
- After Coder responds with a result or error, hand off the result to the Reviewer for validation.
- Relay the final validated result back to the user.
- When you think some question is related to the memory, hand off to the Memory agent, and answer user based on the response, do not assume.
"""
    )

    coder = AssistantAgent(
        name="coder",
        model_client=model_client,
        handoffs=["commander", "reviewer","memory"],
        tools=[move_arm_to_pose, reset_entire_arm_posture, get_moveit_status, move_head, get_enviroment, control_pal_gripper, pick_and_place_by_name, locate_by_name, list_available_objects],
        system_message="""
You are the Coder agent in a multi-agent robot system for the TIAGo robot.

## Role:
- Always plan before action. No plan, no action.
- Execute robot actions (e.g., move, grasp, sense).
- Only act **in response to Commander or Reviewer**.
- Do **not** act autonomously or infer intent.
- When you think some question is related to the memory, hand off to the Memory agent, do not assume.

## You can use:
- `move_arm_to_pose(x, y, z)`
  - Moves the arm to a 3D Cartesian pose. Always validate pose: x ∈ [0.1, 0.8], y ∈ [-0.4, 0.4], z ∈ [0.2, 1.2].
- `control_pal_gripper(joint_positions={...})`
  - To close the gripper:
    `control_pal_gripper(joint_positions={"gripper_left_finger_joint": 0.0, "gripper_right_finger_joint": 0.0})`
  - To open the gripper:
    `control_pal_gripper(joint_positions={"gripper_left_finger_joint": 0.04, "gripper_right_finger_joint": 0.04})`
- `get_moveit_status()`
  - Returns current joint states and end-effector pose.
- `move_head(pan, tilt)`
  - Adjusts the robot's head angle.
- `get_enviroment(prompt="...")`
  - Use this to ask the VLM about object locations in the scene.

## When taking objects:
- Perform grasping using a safe sequence that accounts for the physical offset of the gripper:
  1. Open the gripper.
  2. Move the arm above the object: `move_arm_to_pose(x, y, z + 0.2 + object height)`.
  3. Descend to grasp position: `move_arm_to_pose(x, y, z + 0.2)`.
  4. Close the gripper.
  5. Move the arm up.
"""
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        handoffs=["coder","memory"],
        tools=[],
        system_message="""
You are the Reviewer agent in a multi-agent robot system for the TIAGo robot.

## Role:
- Validate the outputs from the Coder agent.
- Ensure the results are accurate, safe, and meet the user's intent.
- If the output is valid, confirm it to the Commander.
- If the output is invalid, provide detailed feedback to the Coder for correction.
- When you think some question is related to the memory, hand off to the Memory agent, do not assume.

## Guidelines:
- Always check the output format, values, and safety constraints.
- Communicate clearly with the Commander and Coder agents.
- Do not act autonomously; only respond to handoffs from the Commander or Coder.
"""
    )

    memory_agent = AssistantAgent(
        name="memory",
        model_client=model_client,
        tools=[describe_image_from_path, store_triples, query_graph, list_entities, store_image_link, search_visual_memory_from_image, get_entity_by_vector_id],
        handoffs=["commander", "coder", "reviewer"],
        system_message="""
You are a Memory Agent.

Your responsibilities:
- Always begin by calling list_entities() to display all known entities.
- Extract structured triples from user input and store them using store_triples(...).
- Answer memory-related questions by generating Cypher and calling query_graph(...) and display the result.
- When using describe_image_from_path, always provide chat history.
- Always store and retrive as .entity not subject.

Memory Storage Schema:
- All entities are stored in Neo4j as lowercase strings. Match entity names or properties exactly—do not match on labels.
- Use CONTAINS or similar operators if partial matching is required, but always on entity properties, not node labels.

Visual Memory Handling:
- If the user mentions a visual input but does not provide an image path, assume the image is located at /home/haoqi/Desktop/Swarmproject/image/latest.jpg.
- To store a visual memory, call store_image_link(subject, image_path).
- To retrieve the most similar image memory, call search_visual_memory_from_image(image_path).
- If the user asks "Who is this?" or similar questions referring to an image, assume the image path is /home/haoqi/Desktop/Swarmproject/image/latest.jpg and call search_visual_memory_from_image accordingly.

IMPORTANT:
- After storing or answering, always Handoff to the user.
- This ends your turn. Do NOT continue speaking unless the user speaks again.
"""
    )

    team = Swarm(
        [user, commander, coder, reviewer, memory_agent],
        termination_condition=HandoffTermination(target="user")
    )

    # Console input loop
    print("=== TIAGo Multi-Agent Control System ===")
    while True:
        task = input("Enter your instruction (or type 'exit' to quit): ")
        if task.strip().lower() in ["exit", "quit"]:
            break
        await Console(team.run_stream(task=HandoffMessage(source="user", target="commander", content=task)))

    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())


