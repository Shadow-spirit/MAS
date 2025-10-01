#!/usr/bin/env python3
import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

# 1. 加载配置（可换成 configs/alfred_hybrid.yaml 来跑 Hybrid）
config = generic.load_config()
env_type = config['env']['type']  # AlfredTWEnv / AlfredThorEnv / AlfredHybrid

# 2. 创建环境
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# 3. 交互循环
obs, info = env.reset()
print("初始 Obs:", obs[0])

step_id = 0
while True:
    # ---- 打印 admissible commands ----
    admissible = list(info['admissible_commands'][0]) if 'admissible_commands' in info else []
    admissible.sort()
    print(f"\n[Step {step_id}] 可用动作 ({len(admissible)}):")
    for i, a in enumerate(admissible, 1):
        print(f"  {i}. {a}")

    # ---- 用户输入动作 ----
    user = input("> ").strip()
    if user == "":
        if admissible:
            action = np.random.choice(admissible)
            print(f"⚡ 回车检测：随机选择 -> {action}")
        else:
            action = "look around"
    else:
        if admissible and (user not in admissible):
            print(f"⚠️ 注意: 不在 admissible 里，将尝试执行 -> {user}")
        action = user

    # ---- 执行动作 ----
    obs, scores, dones, infos = env.step([action])
    print(f"执行: {action}")
    print("Obs:", obs[0])

    # ---- 打印奖励和评分指标 ----
    print("Reward:", scores[0])
    if isinstance(infos, list) and len(infos) > 0:
        info_dict = infos[0]
        if 'success' in info_dict:
            print("Success:", info_dict['success'])
        if 'goal_condition_success' in info_dict:
            print("GC Success:", info_dict['goal_condition_success'])
        if 'spl' in info_dict:
            print("SPL:", info_dict['spl'])

    info = infos
    step_id += 1
    if dones[0]:
        print(f"✅ Episode finished. Final Score: {scores[0]}")
        break
