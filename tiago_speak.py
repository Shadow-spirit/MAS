#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tiago_say.py — 不在本机安装 sound_play，也能让 TIAGo 说话
做法：通过 SSH 在机器人端执行 `rostopic pub -1 /robotsound sound_play/SoundRequest ...`

用法：
  # 说一句
  ./tiago_say.py --say "Hello, I am TIAGo."

  # 中文
  ./tiago_say.py --say "大家好，我是 TIAGo。"

  # 自定义主机/账号（默认 host=tiago-196c, user=pal, pass=pal）
  ./tiago_say.py --say "Thank you" --host tiago-196c --user pal --password pal

  # 订阅本地 /tiago_say，然后转发到机器人说话（需要本机有 rospy；没有也可省略）
  ./tiago_say.py --listen
  # 另一个终端发布：
  # rostopic pub -1 /tiago_say std_msgs/String "data: 'thank you!'"

前提：
  - 机器人端已安装并能运行 sound_play（多数 TIAGo 已预装）。
  - 机器人上有 `rostopic` 可用，且 `soundplay_node.py` 正在跑（PAL 默认会起）。
"""

import os, sys, json, argparse, subprocess, shlex, time
from typing import Optional

# 可选使用 paramiko（你项目里已有），没有也可走系统 ssh
try:
    import paramiko
except Exception:
    paramiko = None

# 可选导入 rospy，仅在 --listen 时需要
try:
    import rospy
    from std_msgs.msg import String as RosString
except Exception:
    rospy = None
    RosString = None

def build_robotsound_yaml(text: str, volume: float = 1.0) -> str:
    """
    构造 SoundRequest 的 YAML 片段。
    sound_play 常量：SAY=1, PLAY_ONCE=1
    """
    # 用 json.dumps 来安全转义字符串，变成双引号内容
    jtxt = json.dumps(text)
    yaml = f"{{sound: 1, command: 1, arg: {jtxt}, arg2: \"\", volume: {volume:.2f}}}"
    return yaml

def remote_rostopic_pub(host, user, password,
                        text, volume=1.0,
                        topic="/robotsound",
                        ros_distro=None) -> int:
    import shlex, json
    ros_distro = ros_distro or os.getenv("ROS_DISTRO", "noetic")

    # 构造 YAML：外层不用引号；给字符串字段用双引号
    payload = (
        "{sound: 1, command: 1, "
        f'arg: {json.dumps(text)}, '      # -> "thank you!"
        'arg2: "", '
        f"volume: {volume:.2f}"           # 0.00 ~ 1.00
        "}"
    )

    # 关键：在 rostopic 调用里用 **单引号** 包住整段 YAML
    ros_cmd = (
        f"source /opt/ros/{ros_distro}/setup.bash && "
        f"rostopic pub -1 {topic} sound_play/SoundRequest '{payload}'"
    )

    remote_cmd = "bash -lc " + shlex.quote(ros_cmd)

    if paramiko is not None and password:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, port=22, username=user, password=password, timeout=5.0)
            _, stdout, stderr = ssh.exec_command(remote_cmd)
            out = stdout.read().decode("utf-8", "ignore")
            err = stderr.read().decode("utf-8", "ignore")
            rc = stdout.channel.recv_exit_status()
            ssh.close()
            if rc != 0 and err.strip():
                print(f"[remote stderr] {err.strip()}")
            return rc
        except Exception as e:
            print(f"[ssh/paramiko] {e}")
            return 1
    else:
        try:
            return subprocess.call(["ssh", f"{user}@{host}", remote_cmd])
        except Exception as e:
            print(f"[ssh] {e}")
            return 1



def say_once(text: str, volume: float, host: str, user: str, password: Optional[str],
             topic: str, ros_distro: Optional[str]) -> bool:
    rc = remote_rostopic_pub(host, user, password, text, volume, topic, ros_distro)
    if rc == 0:
        print(f"[OK] {text}")
        return True
    print(f"[FAIL rc={rc}] 请确认机器人端已运行 sound_play（soundplay_node.py）且话题 {topic} 有订阅者。")
    return False

def listen_and_forward(host: str, user: str, password: Optional[str],
                       topic_in: str = "/tiago_say", topic_out: str = "/robotsound",
                       volume: float = 1.0, ros_distro: Optional[str] = None):
    if rospy is None:
        print("[warn] 本机无 rospy，无法进入 --listen 模式。")
        return
    if not rospy.core.is_initialized():
        rospy.init_node("tiago_say_forwarder", anonymous=True, disable_signals=True)

    def _cb(msg: "RosString"):
        txt = (msg.data or "").strip()
        if not txt:
            return
        rc = remote_rostopic_pub(host, user, password, txt, volume, topic_out, ros_distro)
        state = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"[forward:{state}] {txt}")

    sub = rospy.Subscriber(topic_in, RosString, _cb, queue_size=10)
    print(f"[listen] 本地订阅 {topic_in} -> 远端 {topic_out} 说话（Ctrl-C 退出）")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()

def main():
    ap = argparse.ArgumentParser(description="Make TIAGo speak via remote rostopic (no local sound_play needed).")
    ap.add_argument("--say", default="", help="要说的话")
    ap.add_argument("--volume", type=float, default=1.0, help="音量 0~1")
    ap.add_argument("--host", default="tiago-196c", help="机器人主机名/IP")
    ap.add_argument("--user", default="pal", help="SSH 用户名")
    ap.add_argument("--password", default="pal", help="SSH 密码（留空则用系统 ssh 免密）")
    ap.add_argument("--topic", default="/robotsound", help="机器人端 TTS 话题")
    ap.add_argument("--ros-distro", default=os.getenv("ROS_DISTRO", "noetic"), help="机器人端 ROS 发行版")
    ap.add_argument("--listen", action="store_true", help="订阅本地 /tiago_say 并转发到机器人说话")
    ap.add_argument("--listen-topic", default="/tiago_say", help="本地订阅的字符串话题")
    args = ap.parse_args()

    if args.say:
        say_once(args.say, args.volume, args.host, args.user, args.password or None, args.topic, args.ros_distro)
        return

    if args.listen:
        listen_and_forward(args.host, args.user, args.password or None,
                           topic_in=args.listen_topic, topic_out=args.topic,
                           volume=args.volume, ros_distro=args.ros_distro)
        return

    # 没参数给提示
    print("示例：")
    print(f"  ./{os.path.basename(__file__)} --say 'Hello, I am TIAGo.'")
    print(f"  ./{os.path.basename(__file__)} --say '谢谢大家。' --host tiago-196c --user pal --password pal")
    print(f"  ./{os.path.basename(__file__)} --listen  # 监听本地 /tiago_say 并转发到机器人 /robotsound")

if __name__ == "__main__":
    main()


