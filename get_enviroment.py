#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

./get_enviroment.py "size=0.10 describe the scene"


"""

import os, sys, json, traceback, re
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose, PoseStamped
import tf2_ros
import tf_conversions
from openai import OpenAI
import cv2

# ================= Config =================


# Topics: 可用环境变量覆盖
RGB_TOPIC     = os.getenv("RGB_TOPIC", "/xtion/rgb/image_raw")
DEPTH_TOPIC   = os.getenv("DEPTH_TOPIC", "/xtion/depth_registered/image_raw")
POINTS_TOPIC  = os.getenv("POINTS_TOPIC", "/xtion/depth_registered/points")
CAMINFO_TOPIC = os.getenv("CAMINFO_TOPIC", "/xtion/rgb/camera_info")

# 输出坐标系（如 TIAGo）
TARGET_FRAME  = os.getenv("TARGET_FRAME", "torso_lift_link")
CAM_OPT_FRAME = os.getenv("CAM_OPT_FRAME", "xtion_rgb_optical_frame")

# ArUco 字典：默认 ORIGINAL，可被 ARUCO_DICT 环境变量覆盖
ARUCO_DICT_NAME = os.getenv("ARUCO_DICT", "DICT_4X4_50")
DEFAULT_MARKER_SIZE_M = os.getenv("MARKER_SIZE_M", "")    # "" 表示不提供
TIMEOUT_SEC = float(os.getenv("ENV_TIMEOUT_SEC", "5.0"))

# ==========================================

# 可选的预定义字典映射（同时支持动态 getattr）
_ARUCO_DICTS = {
    "DICT_4X4_50":    cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100":   cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250":   cv2.aruco.DICT_6X6_250,
    "DICT_7X7_1000":  cv2.aruco.DICT_7X7_1000,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def _get_aruco_dict(dict_name: str):
    # 先尝试 getattr（以便未来 OpenCV 新增常量也能用）
    const = getattr(cv2.aruco, dict_name, None)
    if const is None:
        const = _ARUCO_DICTS.get(dict_name, None)
    if const is None:
        rospy.logwarn(f"[ArUco] Unknown dict '{dict_name}', falling back to DICT_ARUCO_ORIGINAL")
        const = cv2.aruco.DICT_ARUCO_ORIGINAL
    return cv2.aruco.getPredefinedDictionary(const)


class MarkerPose:
    def __init__(self, marker_id:int, center_px, corners_px, pose:Pose=None, tvec_cam=None, rvec_cam=None, size_m=None):
        self.marker_id = int(marker_id)
        self.center_px = center_px
        self.corners_px = corners_px
        self.pose = pose if pose else Pose()
        self.tvec_cam = tvec_cam
        self.rvec_cam = rvec_cam
        self.size_m = size_m

    def to_dict(self):
        out = {
            "id": self.marker_id,
            "center_px": {"x": float(self.center_px[0]), "y": float(self.center_px[1])},
            "corners_px": self.corners_px,
        }
        if self.tvec_cam is not None:
            out["tvec_cam_m"] = {"x": float(self.tvec_cam[0]), "y": float(self.tvec_cam[1]), "z": float(self.tvec_cam[2])}
        if self.rvec_cam is not None:
            out["rvec_cam_rad"] = {"rx": float(self.rvec_cam[0]), "ry": float(self.rvec_cam[1]), "rz": float(self.rvec_cam[2])}
        if self.size_m is not None:
            out["marker_size_m"] = float(self.size_m)
        if self.pose is not None:
            out["pose"] = {
                "position": {
                    "x": self.pose.position.x,
                    "y": self.pose.position.y,
                    "z": self.pose.position.z,
                },
                "orientation": {
                    "x": self.pose.orientation.x,
                    "y": self.pose.orientation.y,
                    "z": self.pose.orientation.z,
                    "w": self.pose.orientation.w,
                },
            }
        return out


def _preprocess_gray(bgr: np.ndarray) -> np.ndarray:
    # 1) 轻度去噪，避免把边缘糊掉
    den = cv2.bilateralFilter(bgr, d=5, sigmaColor=50, sigmaSpace=7)

    # 2) 转 LAB，仅增强亮度通道，减少色偏影响
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 3) 自适应 CLAHE：暗场多一点、亮场少一点
    meanL = float(np.mean(L))
    clip = 2.0 if meanL > 120 else (3.0 if meanL > 80 else 4.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    L = clahe.apply(L)
    lab = cv2.merge([L, A, B])
    enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4) 高光抑制：把接近饱和的像素用邻域中值回填
    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    sat = gray > 245
    if np.any(sat):
        med = cv2.medianBlur(gray, 5)
        gray = np.where(sat, med, gray).astype(np.uint8)

    # 5) 可选：自适应阈值图，供困难场景备用（不直接替换灰度）
    # thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             cv2.THRESH_BINARY, 31, 5)
    return gray


def _wait_one(topic, typ, timeout=TIMEOUT_SEC):
    return rospy.wait_for_message(topic, typ, timeout=timeout)


def _camera_K_from_info(info: CameraInfo):
    K = np.array(info.K, dtype=np.float64).reshape(3,3)
    D = np.array(info.D, dtype=np.float64).reshape(-1,1) if info.D else None
    return K, D


def _parse_marker_size(argv_text: str):
    m = re.search(r"size\s*=\s*([0-9.]+)", argv_text.lower())
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _transform_point_pose(tf_buffer, pose_stamped: PoseStamped, target_frame: str):
    try:
        import tf2_geometry_msgs  # noqa: F401
        transformed = tf_buffer.transform(pose_stamped, target_frame, rospy.Duration(1.0))
        return transformed
    except Exception as e:
        rospy.logwarn(f"[TF] transform pose to {target_frame} failed: {e}")
        return None


def _aruco_detect(rgb_img, K=None, D=None, marker_size_m=None):
    aruco_dict = _get_aruco_dict(ARUCO_DICT_NAME)

    # 灰度预处理（抗光照）
    if len(rgb_img.shape) == 3:
        gray = _preprocess_gray(rgb_img)
    else:
        gray = rgb_img

    # -------- Detector 参数（专门针对 4x4_50 调宽）--------
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    # 阈值窗口更宽
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 123
    params.adaptiveThreshWinSizeStep = 10
    if hasattr(params, "adaptiveThreshConstant"):
        params.adaptiveThreshConstant = 3

    # 允许更小的周长和边界靠近
    if hasattr(params, "minMarkerPerimeterRate"):
        params.minMarkerPerimeterRate = 0.005   # 默认 0.03，放宽到 0.5%
    if hasattr(params, "maxMarkerPerimeterRate"):
        params.maxMarkerPerimeterRate = 4.0
    if hasattr(params, "minCornerDistanceRate"):
        params.minCornerDistanceRate = 0.02
    if hasattr(params, "minDistanceToBorder"):
        params.minDistanceToBorder = 1

    # 亚像素精修更耐心
    if hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "cornerRefinementWinSize"):
        params.cornerRefinementWinSize = 7
    if hasattr(params, "cornerRefinementMaxIterations"):
        params.cornerRefinementMaxIterations = 60
    if hasattr(params, "cornerRefinementMinAccuracy"):
        params.cornerRefinementMinAccuracy = 0.001

    # 允许黑白反转（反光时有用）
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True

    # OpenCV 新版 aruco3 更鲁棒
    if hasattr(params, "useAruco3Detection"):
        params.useAruco3Detection = True

    # -------- 检测 --------
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    # Debug 可视化
    dbg = rgb_img.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
    else:
        # 保存预处理灰度以便排查
        cv2.imwrite("/tmp/aruco_gray.jpg", gray)
    cv2.imwrite("/tmp/aruco_debug.jpg", dbg)

    # -------- 组装返回 --------
    markers = []
    if ids is None or len(ids) == 0:
        return markers

    ids = ids.flatten()
    rvecs, tvecs = [None]*len(ids), [None]*len(ids)

    if (K is not None) and (marker_size_m is not None):
        half = marker_size_m/2.0
        objp = np.array(
            [[-half,  half, 0],
             [ half,  half, 0],
             [ half, -half, 0],
             [-half, -half, 0]], dtype=np.float32
        )
        for i, mid in enumerate(ids):
            ok, rvec, tvec = cv2.solvePnP(
                objp, corners[i][0].astype(np.float32),
                K, D if D is not None else None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                rvecs[i] = rvec.reshape(-1)
                tvecs[i] = tvec.reshape(-1)

    for i, mid in enumerate(ids):
        ctr = np.mean(corners[i][0], axis=0).tolist()
        tvec = tvecs[i].tolist() if tvecs[i] is not None else None
        rvec = rvecs[i].tolist() if rvecs[i] is not None else None
        mk = MarkerPose(
            marker_id=int(mid),
            center_px=ctr,
            corners_px=corners[i][0].tolist(),
            tvec_cam=tvec, rvec_cam=rvec, size_m=marker_size_m
        )
        markers.append(mk)

    return markers


def _vlm_describe(rgb_bgr: np.ndarray, depth_gray: np.ndarray):
    if not os.getenv("OPENAI_API_KEY"):
        return ""

    import base64
    rgb_path = "/tmp/_rgb.jpg"
    depth_path = "/tmp/_depth.jpg"
    cv2.imwrite(rgb_path, rgb_bgr)
    depth_vis = cv2.normalize(np.nan_to_num(depth_gray, nan=0.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(depth_path, depth_bgr)

    with open(rgb_path, "rb") as f1, open(depth_path, "rb") as f2:
        rgb_b64 = base64.b64encode(f1.read()).decode()
        depth_b64 = base64.b64encode(f2.read()).decode()

    client = OpenAI()
    prompt_text = (
        "You are a robot vision assistant. Given an RGB image and a depth visualization, "
        "write a short natural-language description of the scene (1-2 sentences). "
        "Do not return JSON. Do not list objects; just describe the scene."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{depth_b64}"}},
                ]
            }]
        )
        txt = resp.choices[0].message.content.strip()
        return txt
    except Exception as e:
        rospy.logwarn(f"[VLM] description failed: {e}")
        return ""


def get_enviroment(prompt: str) -> dict:
    """
    混合输出：
    - ArUco 返回 marker（像素与可选的位姿）
    - VLM 返回简短文本描述（可选）
    输出 JSON:
      {
        "description": "...",
        "markers": [ {id, center_px, corners_px, tvec_cam_m?, rvec_cam_rad?, marker_size_m?, pose?} ],
        "objects": []
      }
    """
    try:
        try:
            rospy.init_node("get_env_node", anonymous=True)
        except rospy.exceptions.ROSException:
            pass

        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)  # noqa: F841
        bridge = CvBridge()

        # 读取图像/深度/点云/相机内参
        rgb_msg   = _wait_one(RGB_TOPIC, Image, TIMEOUT_SEC)
        depth_msg = None
        try:
            depth_msg = _wait_one(DEPTH_TOPIC, Image, 1.0)
        except Exception:
            pass
        points_msg = None
        try:
            points_msg = _wait_one(POINTS_TOPIC, PointCloud2, 1.0)
        except Exception:
            pass
        info_msg = None
        K = D = None
        try:
            info_msg = _wait_one(CAMINFO_TOPIC, CameraInfo, 1.0)
            K, D = _camera_K_from_info(info_msg)
        except Exception:
            pass

        # 转 cv2 / numpy
        rgb_bgr = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        if depth_msg is not None:
            try:
                depth_gray = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            except Exception:
                depth_gray = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1").astype(np.float32)
        else:
            depth_gray = np.zeros_like(rgb_bgr[:,:,0]).astype(np.float32)

        # 1) 场景描述（可选）
        description = _vlm_describe(rgb_bgr, depth_gray)

        # 2) ArUco 检测 + 位姿
        argv_text = prompt if isinstance(prompt, str) else ""
        size_m = _parse_marker_size(argv_text)
        if size_m is None and DEFAULT_MARKER_SIZE_M.strip():
            try:
                size_m = float(DEFAULT_MARKER_SIZE_M)
            except Exception:
                size_m = None

        markers = _aruco_detect(rgb_bgr, K=K, D=D, marker_size_m=size_m)

        # 3) 变换到 TARGET_FRAME（如果有 tvec_cam 或点云）
        out_markers = []
        for mk in markers:
            pose_cam = PoseStamped()
            pose_cam.header.stamp = rospy.Time(0)
            pose_cam.header.frame_id = CAM_OPT_FRAME

            if mk.tvec_cam is not None:
                pose_cam.pose.position.x = float(mk.tvec_cam[0])
                pose_cam.pose.position.y = float(mk.tvec_cam[1])
                pose_cam.pose.position.z = float(mk.tvec_cam[2])
            elif points_msg is not None:
                cx, cy = mk.center_px
                try:
                    gen = pc2.read_points(points_msg, field_names=("x","y","z"), uvs=[(int(cx), int(cy))], skip_nans=True)
                    p = next(gen, None)
                    if p is not None:
                        pose_cam.pose.position.x, pose_cam.pose.position.y, pose_cam.pose.position.z = map(float, p[:3])
                        pose_cam.header.frame_id = points_msg.header.frame_id
                except Exception:
                    pass

            pose_cam.pose.orientation.w = 1.0
            pose_out = _transform_point_pose(tf_buffer, pose_cam, TARGET_FRAME)
            if pose_out is None:
                pose_out = pose_cam  # 至少返回当前坐标系

            mk.pose = pose_out.pose
            out_markers.append(mk.to_dict())

        return {
            "description": description,
            "markers": out_markers,
            "objects": []
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Processing failed: {e}"}


if __name__ == "__main__":
    prm = sys.argv[1] if len(sys.argv) > 1 else "size=0.045 describe the scene"
    res = get_enviroment(prm)
    print(json.dumps(res, ensure_ascii=False, indent=2))

