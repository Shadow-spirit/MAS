#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gc_refine.py — Drop-in helper for secondary XY refinement using a gripper camera.

How to use in your ik_to_marker.py:
-----------------------------------
1) Add near the imports:
    from gc_refine import refine_xy_with_gripper_cam, GCConfig

2) Before you descend (在下降到z_place之前), call:
    x_ref, y_ref, seed_map = refine_xy_with_gripper_cam(
        ik=ik,
        arm_pub=arm_pub,
        seed_map=seed_map,           # your current seed (dict of joint -> pos)
        x_tgt=x_tgt, y_tgt=y_tgt,    # current hover XY
        z_hover=z_hover,             # current hover Z
        yaw_hint=your_yaw_hint,      # or 0.0
        cfg=GCConfig(                 # 可按需调整参数
            img_cx=320,
            img_cy=240,
            scale_x_m_per_px=0.0009,
            scale_y_m_per_px=0.0009,
            thresh_px=8,
            max_iters=5,
            shrink=0.7,
            sign_x=-1.0,
            sign_y=+1.0,
            topic="/gripper_cam_aruco_pose"
        )
    )
    # 然后用 (x_ref, y_ref) 继续你的下降阶段 IK

This file intentionally has *zero* dependencies on your original code structure,
except that you expose two callables:
  - solve_ik_at(ik, x, y, z, seed_map, yaw_hint=0.0) -> (qmap or None, rpy or None)
  - send_arm(arm_pub, qmap, duration)
and you provide a valid "ik" (trac_ik_python.IK) and "arm_pub" (JointTrajectory publisher).

If your function names differ, simply wrap them before calling this helper.
"""

import rospy
from geometry_msgs.msg import PointStamped
import threading

# ===================== Public Config =====================
class GCConfig(object):
    def __init__(self,
                 img_cx=320, img_cy=240,
                 scale_x_m_per_px=0.0009, scale_y_m_per_px=0.0009,
                 thresh_px=8, max_iters=5, shrink=0.7,
                 sign_x=-1.0, sign_y=+1.0,
                 topic="/gripper_cam_aruco_pose"):
        self.img_cx = float(img_cx)
        self.img_cy = float(img_cy)
        self.scale_x_m_per_px = float(scale_x_m_per_px)
        self.scale_y_m_per_px = float(scale_y_m_per_px)
        self.thresh_px = float(thresh_px)
        self.max_iters = int(max_iters)
        self.shrink = float(shrink)
        self.sign_x = float(sign_x)
        self.sign_y = float(sign_y)
        self.topic = str(topic)

# ===================== Internal subscriber buffer =====================
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

# ===================== Main helper =====================
def refine_xy_with_gripper_cam(ik,
                               arm_pub,
                               seed_map,
                               x_tgt, y_tgt, z_hover,
                               yaw_hint=0.0,
                               cfg=GCConfig(),
                               solve_ik_at=None,
                               send_arm=None,
                               move_time_fast=1.2):
    """
    Returns: (x_refined, y_refined, updated_seed_map)
    Required keyword callables if names differ in your code:
      - solve_ik_at: function(ik, x, y, z, seed_map, yaw_hint) -> (qmap or None, rpy or None)
      - send_arm:    function(arm_pub, qmap, duration)
    """
    if solve_ik_at is None:
        raise RuntimeError("refine_xy_with_gripper_cam requires solve_ik_at callable")
    if send_arm is None:
        raise RuntimeError("refine_xy_with_gripper_cam requires send_arm callable")

    buf = _GCPoseBuffer(cfg.topic)
    rospy.loginfo("[GC] Start XY refinement using %s at z=%.3f", cfg.topic, z_hover)

    x_cur, y_cur = float(x_tgt), float(y_tgt)
    step_scale = 1.0
    seed_local = dict(seed_map)

    for it in range(cfg.max_iters):
        # wait up to 1s for observation
        t0 = rospy.Time.now()
        obs = None
        while (rospy.Time.now() - t0).to_sec() < 1.0:
            obs = buf.get()
            if obs is not None:
                break
            rospy.sleep(0.02)
        if obs is None:
            rospy.logwarn("[GC] No observation in iter %d; stop refinement.", it+1)
            break

        u = float(obs.point.x)
        v = float(obs.point.y)
        du = u - cfg.img_cx
        dv = v - cfg.img_cy
        rospy.loginfo("[GC] iter=%d uv=(%.1f,%.1f) du=%.1f dv=%.1f", it+1, u, v, du, dv)

        # within pixel threshold
        if abs(du) <= cfg.thresh_px and abs(dv) <= cfg.thresh_px:
            rospy.loginfo("[GC] Pixel error within threshold; stop refinement.")
            break

        # pixel -> meters with configurable signs
        dx = cfg.sign_x * (-du) * cfg.scale_x_m_per_px * step_scale
        dy = cfg.sign_y * (+dv) * cfg.scale_y_m_per_px * step_scale

        x_new = x_cur + dx
        y_new = y_cur + dy

        sol, _ = solve_ik_at(ik, x_new, y_new, z_hover, seed_local, yaw_hint=yaw_hint)
        if not sol:
            rospy.logwarn("[GC] IK failed at iter %d (x=%.3f,y=%.3f); shrink step and continue.", it+1, x_new, y_new)
            step_scale *= cfg.shrink
            continue

        send_arm(arm_pub, sol, duration=move_time_fast)
        rospy.sleep(0.2)

        # update seed and current target
        seed_local.update(sol)
        x_cur, y_cur = x_new, y_new
        step_scale *= cfg.shrink

    return x_cur, y_cur, seed_local
