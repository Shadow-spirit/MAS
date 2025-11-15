#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
IK to ArUco marker (TIAGo) + optional Gripper-Cam secondary XY refinement
（已整合：保留原功能，修复输入与 Pose 附加属性问题，加入自动拉起 gripper_cam）

支持 pick/place 两种模式:
- pick: 抓取 (末端下降 → close gripper)
- place: 放置 (末端下降 → open gripper)

新增/修复要点：
1) 选择 Head Cam 的 marker ID 后，自动以相同 ID 后台启动本地 gripper_cam.py；
2) “阶段1: hover XY”与“阶段2: descent”之间进行夹爪相机二次校准（可关）；
3) fetch_marker_target_pose() 采用**稳健输入解析**（忽略方向键转义等非数字字符），并**返回 (Pose, chosen_id)**；
4) 二次校准等待像素观测放宽到 3s，更稳。
"""

import os, sys, re, json, math, time, subprocess, threading, shlex
import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import Pose, PointStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from trac_ik_python.trac_ik import IK


ID_NAME_MAP = {
    1: "block1",
    2: "block2",
    10: "roof1"
}
# ========= 配置 =========
MARKER_CMD        = ["python3", "get_enviroment.py", "size=0.045 describe the scene"]
TARGET_ID         = None
BASE_LINK         = "torso_lift_link"
EE_LINK           = "gripper_link"
ARM_CONTROLLER    = "/arm_controller/command"

ARM_JOINTS = [
    "arm_1_joint","arm_2_joint","arm_3_joint","arm_4_joint",
    "arm_5_joint","arm_6_joint","arm_7_joint"
]

# IK/搜索策略
SOLVE_ATTEMPTS   = 10
YAW_SAMPLES      = 16
ROLLPITCH_TOL    = 0.05
XY_RADIUS_CLAMP  = 0.70

# Z 相关
EEF_TO_GRIPPER_OFFSET = 0.22   # end-effector → gripper 补偿
PLACE_EXTRA_BIAS      = 0.05
HOVER_UP              = 0.10   # hover 高度
Z_DESCENT_STEPS       = 8
APPROACH_DURATION     = 3.0
DESCENT_DURATION      = 4.0

# 夹爪
GRIPPER_TOPIC   = "/gripper_controller/follow_joint_trajectory/goal"
GRIPPER_OPEN    = (0.04, 0.04)
GRIPPER_CLOSE   = (0.00, 0.00)

# ========= 夹爪相机二次校准配置 =========
ENABLE_GRIPPER_CAM_REFINE = True
GRIPPER_CAM_TOPIC         = "/gripper_cam_aruco_pose"   # geometry_msgs/PointStamped, point.x=u, point.y=v
GC_IMG_CX                 = 320
GC_IMG_CY                 = 240

# ——像素->米比例：改成毫米级——
GC_SCALE_X_M_PER_PX       = 0.00020   # 原 0.0004/0.0009 太大 → 0.2 mm/px
GC_SCALE_Y_M_PER_PX       = 0.00020

GC_THRESH_PX              = 8
GC_MAX_ITERS              = 4         # 减少迭代，稳定为先
GC_SHRINK                 = 0.6       # 衰减快一点，小碎步更明显
GC_SIGN_X                 = -1.0
GC_SIGN_Y                 = +1.0

# ——硬限幅：每步 ≤ 4mm；整个过程 ≤ 10mm——
GC_MAX_STEP_M             = 0.004     # 4 mm/step（防止单步过大）
GC_MAX_TOTAL_M            = 0.010     # 10 mm total（防止累计跑飞）


# ========= 工具 =========
def quat_from_two_vectors(v1, v2):
    v1 = np.asarray(v1, dtype=float); n1 = np.linalg.norm(v1)
    v2 = np.asarray(v2, dtype=float); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return [0,0,0,1]
    v1 /= n1; v2 /= n2
    c = np.cross(v1, v2); d = float(np.dot(v1, v2))
    if d < -0.999999:
        axis = np.array([1.0,0.0,0.0]) if abs(v1[0]) < 0.9 else np.array([0.0,1.0,0.0])
        return tft.quaternion_about_axis(math.pi, axis).tolist()
    s = math.sqrt((1.0 + d) * 2.0)
    q = np.array([c[0]/s, c[1]/s, c[2]/s, s/2.0])
    return q.tolist()

def get_current_joint_positions(names):
    js = rospy.wait_for_message("/joint_states", JointState, timeout=2.0)
    name_to_pos = dict(zip(js.name, js.position))
    return [name_to_pos.get(n, 0.0) for n in names]

def get_current_ee(tf_buffer):
    trans = tf_buffer.lookup_transform(BASE_LINK, EE_LINK, rospy.Time(0), rospy.Duration(1.0))
    return float(trans.transform.translation.x), float(trans.transform.translation.y), float(trans.transform.translation.z)

def reorder_to_arm_joints(ik_joint_names, ik_solution):
    name_to_val = dict(zip(ik_joint_names, ik_solution))
    return [name_to_val[j] for j in ARM_JOINTS]

def publish_arm_single(pub, q_to, duration=5.0):
    traj = JointTrajectory()
    traj.joint_names = ARM_JOINTS
    pt = JointTrajectoryPoint()
    pt.positions = q_to
    pt.time_from_start = rospy.Duration(float(duration))
    traj.points = [pt]
    pub.publish(traj)

def publish_arm_stepwise(pub, points, dt_each):
    for q in points:
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.time_from_start = rospy.Duration(float(dt_each))
        traj.points = [pt]
        pub.publish(traj)
        rospy.sleep(dt_each + 0.05)

def gripper_move(mode="open", duration=2.0):
    pos = GRIPPER_OPEN if mode=="open" else GRIPPER_CLOSE
    cmd = [
        "rostopic","pub","-1",
        GRIPPER_TOPIC,"control_msgs/FollowJointTrajectoryActionGoal",
        f"goal: {{ trajectory: {{ joint_names: ['gripper_left_finger_joint','gripper_right_finger_joint'], "
        f"points: [ {{ positions: [{pos[0]},{pos[1]}], time_from_start: {{secs: {int(duration)}}} }} ] }} }}"
    ]
    subprocess.run(cmd, text=True)
    rospy.loginfo("Gripper %s", mode)

# ========= JSON 读取 =========
def _run_get_env_json(cmd=MARKER_CMD, timeout=20):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, timeout=timeout)
    sout = proc.stdout.decode("utf-8", errors="ignore")
    m = re.search(r"\{.*\}\s*$", sout, flags=re.S)
    if not m: return None
    return json.loads(m.group(0))

def fetch_marker_target_pose():
    data = _run_get_env_json()
    if not data or not data.get("markers"):
        raise RuntimeError("No markers found")

    markers = data["markers"]

    print("\nDetected markers:")
    for m in markers:
        pos = m["pose"]["position"]
        mid = int(m["id"])
        name = ID_NAME_MAP.get(mid)
        name_str = f" ({name})" if name else ""
        print(f"  ID {mid}{name_str}: x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}")

    # -------- 稳健输入：只提取整数ID，忽略方向键/Home/End等转义 --------
    while True:
        raw = input("Enter target marker ID (or multiple IDs for PLACE, e.g. '1 2 5', or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit(0)
        toks = re.findall(r"-?\d+", raw)   # 只抓整数，过滤掉 ^[[F 等
        ids = [int(t) for t in toks]
        if ids:
            break
        print("No valid integer IDs detected, please try again...")

    # 选第一个ID作为“基准ID”（X/Z 取它；Y 可在place模式下用多ID的中位数）
    chosen = ids[0]
    m = next((mm for mm in markers if int(mm["id"]) == chosen), None)
    if m is None:
        raise RuntimeError(f"Marker ID {chosen} not found.")

    # 计算 z 偏置（沿用你的全局 mode、EEF_TO_GRIPPER_OFFSET、PLACE_EXTRA_BIAS）
    if mode == "pick":
        z_bias = EEF_TO_GRIPPER_OFFSET
    else:  # place
        z_bias = EEF_TO_GRIPPER_OFFSET + PLACE_EXTRA_BIAS

    # ---- 基准位姿（X/Z 来自 chosen）----
    pos0 = m["pose"]["position"]
    x0 = float(pos0["x"])
    y0 = float(pos0["y"])
    z0 = float(pos0["z"])

    # ---- 如果是 place 且输入了多个ID，就把 Y 设为这些ID的中位数 ----
    if mode == "place" and len(ids) > 1:
        ys = []
        missing = []
        for tid_i in ids:
            mm = next((mm for mm in markers if int(mm["id"]) == tid_i), None)
            if mm is None:
                missing.append(tid_i)
                continue
            ys.append(float(mm["pose"]["position"]["y"]))
        if ys:
            y0 = float(np.median(ys))
            if missing:
                rospy.logwarn("Some IDs not found (ignored): %s", ",".join(map(str, missing)))
        else:
            rospy.logwarn("No valid IDs for Y-median; fallback to chosen ID's Y")

    # ---- 应用你的 XY 偏置与 Z 补偿 ----
    tgt = Pose()
    tgt.position.x = x0 - 0.02
    tgt.position.y = (y0 - 0.01) if (y0 > 0.0) else (y0 + 0.01)
    tgt.position.z = z0 + z_bias
    tgt.orientation.w = 1.0

    rospy.loginfo(
        "Target (base ID=%d): x=%.3f y=%.3f z=%.3f (+EEF=%.3fm%s)",
        chosen, tgt.position.x, tgt.position.y, tgt.position.z,
        EEF_TO_GRIPPER_OFFSET,
        f", +PLACE_EXTRA={PLACE_EXTRA_BIAS:.3f}m" if mode == "place" else ""
    )
    return tgt, chosen   # 返回二元组

# ========= IK =========
def quat_palm_down(yaw=0.0):
    v_tool = np.array([0,0,-1.0])
    v_base_down = np.array([0,0,-1.0])
    q_align = quat_from_two_vectors(v_tool, v_base_down)
    q_yaw   = tft.quaternion_about_axis(yaw, v_base_down)
    return tft.quaternion_multiply(q_yaw, q_align)

def solve_ik_at(ik, x, y, z, seed_map, yaw_hint=None):
    pos_tol = (0.01,0.01,0.01)
    rot_tol = (ROLLPITCH_TOL,ROLLPITCH_TOL,math.pi)
    yaw_list = [yaw_hint] if yaw_hint is not None else np.linspace(-math.pi,math.pi,YAW_SAMPLES)

    r = math.hypot(x,y)
    if r > XY_RADIUS_CLAMP and r > 1e-6:
        s = XY_RADIUS_CLAMP / r
        x, y = x*s, y*s

    for yaw in yaw_list:
        qx,qy,qz,qw = quat_palm_down(yaw)
        seed = [seed_map.get(n,0.0) for n in ik.joint_names]
        for k in range(SOLVE_ATTEMPTS):
            sol = ik.get_ik(seed,x,y,z,qx,qy,qz,qw,*pos_tol,*rot_tol)
            if sol: return sol,yaw
            seed = [s+0.01*math.sin(0.7*k+i) for i,s in enumerate(seed)]
    return None,None

# ========= 夹爪相机微调实现 =========
class _GCPoseBuffer(object):
    def __init__(self, topic):
        self._lock = threading.Lock()
        self._last = None
        self._sub  = rospy.Subscriber(topic, PointStamped, self._cb, queue_size=1)
    def _cb(self, msg):
        with self._lock:
            self._last = msg
    def get(self):
        with self._lock:
            return self._last

def refine_xy_with_gripper_cam(ik, arm_pub, seed_map, x_tgt, y_tgt, z_hover, yaw_hint=None):
    """
    在固定 z_hover 下，用夹爪相机像素偏差做“毫米级小碎步” XY 微调。
    - 先等首帧（最多 5s），没拿到 → 跳过（不影响主流程）
    - 每次步长会按 GC_SHRINK 衰减
    - 每步/总步位移都有硬限幅
    - 若像素变化 < 2px 连续两次，提前停止
    返回 (x_ref, y_ref, updated_seed_map)；seed_map 的键使用 ik.joint_names。
    """
    if not ENABLE_GRIPPER_CAM_REFINE:
        return x_tgt, y_tgt, seed_map

    buf = _GCPoseBuffer(GRIPPER_CAM_TOPIC)
    rospy.loginfo("[GC] Secondary XY refinement start at z=%.3f", z_hover)

    # ——等待首帧（5s 预热），避免“还没看到就开始”的情况——
    t0 = rospy.Time.now()
    first = None
    while (rospy.Time.now() - t0).to_sec() < 5.0:
        first = buf.get()
        if first is not None:
            break
        rospy.sleep(0.05)
    if first is None:
        rospy.logwarn("[GC] No gripper-cam obs (warmup). Skip refinement.")
        return x_tgt, y_tgt, seed_map

    x_cur, y_cur = float(x_tgt), float(y_tgt)
    step_scale   = 1.0
    seed_local   = dict(seed_map)
    total_dx = 0.0
    total_dy = 0.0
    last_du = last_dv = None

    for it in range(GC_MAX_ITERS):
        # ——单次迭代内，最多再等 3s，拿一条观测——
        t1 = rospy.Time.now()
        obs = None
        while (rospy.Time.now() - t1).to_sec() < 3.0:
            obs = buf.get()
            if obs is not None:
                break
            rospy.sleep(0.03)
        if obs is None:
            rospy.logwarn("[GC] No obs on %s (iter %d).", GRIPPER_CAM_TOPIC, it+1)
            break

        u = float(obs.point.x)
        v = float(obs.point.y)
        du = u - GC_IMG_CX
        dv = v - GC_IMG_CY
        rospy.loginfo("[GC] iter=%d uv=(%.1f,%.1f) du=%.1f dv=%.1f", it+1, u, v, du, dv)

        # ——像素变化太小：提前收敛——
        if last_du is not None and abs(du - last_du) < 2.0 and abs(dv - last_dv) < 2.0:
            rospy.loginfo("[GC] Pixel change < 2px twice; stop refinement.")
            break
        last_du, last_dv = du, dv

        # ——已在阈值内：结束——
        if abs(du) <= GC_THRESH_PX and abs(dv) <= GC_THRESH_PX:
            rospy.loginfo("[GC] Pixel within threshold; stop refinement.")
            break

        # ——像素->米 + 衰减——
        dx = GC_SIGN_X * (-du) * GC_SCALE_X_M_PER_PX * step_scale
        dy = GC_SIGN_Y * (+dv) * GC_SCALE_Y_M_PER_PX * step_scale

        # ——每步硬限幅（防止单步大甩）——
        dx = max(min(dx, GC_MAX_STEP_M), -GC_MAX_STEP_M)
        dy = max(min(dy, GC_MAX_STEP_M), -GC_MAX_STEP_M)

        # ——累计硬限幅（防止越走越远）——
        next_total_dx = total_dx + dx
        next_total_dy = total_dy + dy
        if abs(next_total_dx) > GC_MAX_TOTAL_M:
            dx = math.copysign(max(0.0, GC_MAX_TOTAL_M - abs(total_dx)), dx)
            next_total_dx = total_dx + dx
        if abs(next_total_dy) > GC_MAX_TOTAL_M:
            dy = math.copysign(max(0.0, GC_MAX_TOTAL_M - abs(total_dy)), dy)
            next_total_dy = total_dy + dy

        x_new = x_cur + dx
        y_new = y_cur + dy

        sol, _ = solve_ik_at(ik, x_new, y_new, z_hover, seed_local, yaw_hint=yaw_hint)
        if not sol:
            rospy.logwarn("[GC] IK failed at iter %d (x=%.3f,y=%.3f); shrink step.", it+1, x_new, y_new)
            step_scale *= GC_SHRINK
            continue

        # ——下发小步动作（短时长，让动作更“轻”）——
        q_cmd = reorder_to_arm_joints(ik.joint_names, sol)
        publish_arm_single(arm_pub, q_cmd, duration=0.8)
        rospy.sleep(0.15)

        # 更新 seed / 状态
        seed_local = dict(zip(ik.joint_names, sol))
        x_cur, y_cur = x_new, y_new
        total_dx, total_dy = next_total_dx, next_total_dy
        step_scale *= GC_SHRINK

    return x_cur, y_cur, seed_local


# ========= 主流程 =========
if __name__ == "__main__":
    if len(sys.argv)<2 or sys.argv[1] not in ("pick","place"):
        print("Usage: python3 ik_to_marker.py [pick|place]")
        sys.exit(1)
    mode = sys.argv[1]

    rospy.init_node("ik_to_marker_with_hover", anonymous=True)
    arm_pub = rospy.Publisher(ARM_CONTROLLER, JointTrajectory, queue_size=2)
    rospy.sleep(0.5)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    ik = IK(BASE_LINK, EE_LINK)

    # ===== 获取目标并（可选）启动夹爪相机脚本 =====
    target, chosen_id = fetch_marker_target_pose()
    x_tgt, y_tgt, z_tgt = target.position.x, target.position.y, target.position.z



    q_now = get_current_joint_positions(ARM_JOINTS)
    seed_map = dict(zip(ARM_JOINTS,q_now))

    try:
        _,_,z_now = get_current_ee(tf_buffer)
    except:
        z_now = z_tgt

    # 阶段0: hover 提升
    z_hover = z_tgt + HOVER_UP
    if z_now < z_hover-1e-3:
        x_now,y_now,_ = get_current_ee(tf_buffer)
        sol_lift,_ = solve_ik_at(ik,x_now,y_now,z_hover,seed_map)
        if sol_lift:
            q_hover0 = reorder_to_arm_joints(ik.joint_names,sol_lift)
            publish_arm_single(arm_pub,q_hover0,duration=APPROACH_DURATION*0.7)
            rospy.sleep(APPROACH_DURATION*0.7+0.1)
            seed_map = dict(zip(ARM_JOINTS,q_hover0))
    if mode=="pick":
        gripper_move("open",duration=2.0)
    # 阶段1: hover XY
    sol1,yaw_used = solve_ik_at(ik,x_tgt,y_tgt,z_hover,seed_map)
    if not sol1: sys.exit("XY IK failed")
    q_xy_hover = reorder_to_arm_joints(ik.joint_names,sol1)
    publish_arm_single(arm_pub,q_xy_hover,duration=APPROACH_DURATION)
    rospy.sleep(APPROACH_DURATION+0.1)



    # 阶段1.5: 夹爪相机二次校准（可选）
    # try:
    #     if chosen_id is not None:
    #         cmd = f"python3 gripper_cam.py {chosen_id} --image-topic /camera/image_raw --dict DICT_4X4_100"

    #         subprocess.Popen(shlex.split(cmd))   # 后台启动本地脚本（按需改路径）
    #         rospy.loginfo("[GC] Launched local gripper_cam.py with target_id=%s", chosen_id)
    # except Exception as e:
    #     rospy.logwarn("[GC] Failed to launch gripper-cam: %s", str(e))
    try:
        x_ref, y_ref, seed_map_ref = refine_xy_with_gripper_cam(
            ik=ik,
            arm_pub=arm_pub,
            seed_map=dict(zip(ik.joint_names, sol1)),  # 以 ik.joint_names 顺序作为 seed
            x_tgt=x_tgt, y_tgt=y_tgt,
            z_hover=z_hover,
            yaw_hint=yaw_used
        )
        x_used, y_used = x_ref, y_ref
        seed_base = seed_map_ref  # 以 ik.joint_names 保存
        rospy.loginfo("[GC] Refined XY: x=%.3f y=%.3f", x_used, y_used)
    except Exception as e:
        rospy.logwarn("[GC] refinement skipped due to error: %s", str(e))
        x_used, y_used = x_tgt, y_tgt
        seed_base = dict(zip(ik.joint_names, sol1))

    # 阶段2: 下降（沿用 refine 后的 XY；若跳过 refine，则仍用原 XY）
    zs = list(np.linspace(z_hover,z_tgt,Z_DESCENT_STEPS+1)[1:])
    seed_map2 = seed_base  # keys = ik.joint_names
    points=[]
    for zi in zs:
        sol_i,_ = solve_ik_at(ik,x_used,y_used,zi,seed_map2,yaw_hint=yaw_used)
        if not sol_i: sys.exit(f"Descent IK fail at {zi}")
        q_i = reorder_to_arm_joints(ik.joint_names,sol_i)
        points.append(q_i)
        seed_map2 = dict(zip(ik.joint_names, sol_i))

    publish_arm_stepwise(arm_pub,points,dt_each=DESCENT_DURATION/max(1,len(zs)))

    # 夹爪动作
    if mode=="pick":
        gripper_move("close",duration=2.0)
    else: # place
        gripper_move("open",duration=2.0)

    rospy.sleep(2.5)

    # 回 hover
    sol_hover_back,_ = solve_ik_at(ik,x_used,y_used,z_hover,seed_map2,yaw_hint=yaw_used)
    if sol_hover_back:
        q_hover_back = reorder_to_arm_joints(ik.joint_names,sol_hover_back)
        publish_arm_single(arm_pub,q_hover_back,duration=APPROACH_DURATION)
        rospy.sleep(APPROACH_DURATION+0.2)

    # 回 home
    q_return=[0.1,0.4,-1.41,1.71,0.43,-1.37,1.7]
    publish_arm_single(arm_pub,q_return,duration=5.0)
    rospy.loginfo("Done (%s). Returned to home.",mode)
