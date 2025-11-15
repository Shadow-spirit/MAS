#!/usr/bin/env python3.10
import asyncio
import json
import ast
import os
import csv
from scienceworld import ScienceWorldEnv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ====== 环境封装 ======
class ScienceWorldWrapper:
    def __init__(self, task_name="chemistry-mix-paint-secondary-color", var_num=0, step_limit=10):
        self.env = ScienceWorldEnv("", None, envStepLimit=step_limit)
        self.env.load(task_name, var_num,
                      simplificationStr="teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction",
                      generateGoldPath=True)
        self.episode_actions = []
        self.task_name = task_name

    def reset(self):
        obs, info = self.env.reset()
        self.episode_actions = []
        return {
            "obs": obs,
            "valid_actions": self.env.get_possible_actions(),
            "actions": list(self.episode_actions),
            "goal": info.get("taskDesc")
        }

    def step(self, action: str):
        obs, reward, done, info = self.env.step(action)
        self.episode_actions.append(action)
        return {
            "obs": obs,
            "reward": reward,
            "done": done,
            "score": info.get("score", 0),
            "valid_actions": self.env.get_possible_actions(),
            "actions": list(self.episode_actions)
        }


# ====== 单智能体 ======
def build_agent(client, env: ScienceWorldWrapper):
    user = UserProxyAgent(name="user")

    solo_agent = AssistantAgent(
        name="solo",
        model_client=client,
        handoffs=["user"],
        tools=[env.reset, env.step],
        system_message="""
You are a Single Agent responsible for completing the ScienceWorld task by yourself.

Responsibilities:
- Use env.reset() to start the environment.
- Carefully read the goal from the observation.
- Then, in each step, use env.step(action) to perform exactly ONE valid action at a time.
- Only select actions that appear in the valid_actions list.
- Explain your reasoning briefly before each step.
- Keep trying until the task is done (done=True).
- When the task is finished, summarize the entire process clearly.


CRITICAL RULES:
- You must ONLY select an action string that matches EXACTLY one of the templates in the current valid_actions list.
- Replace "OBJ" in the template with the exact object names shown in the observation.
- Do not invent verbs or action formats.
- Do not repeat actions.
- mix OBJ mean mix two OBJ in same container. pour OBJ into OBJ mean pour one OBJ into another OBJ.
- If object contain something, use this format, "<action> <container> containing <OBJ>". No brackets.
- For multiple containing "<action> <container> containing <OBJ1> and <OBJ2>". Link with "and". Also no brackets.
- look in action allow you to check what in container.
Important : When the final goal is achieved, you must focus on the final product itself.Do NOT focus on the container.(e.g focus on wood cup containing orange juice -> focus on orange juice)
- When you need an object name, use look around to observe.

Termination rule:
- When done=True in env.step, summarize the full episode:
  - Step-by-step actions and their outcomes
  - Total score and number of moves
  - Then end the conversation (handoff to user).
- If moves >= 20 also handoff to user.

"""
    )

    return [user, solo_agent]


# ====== 单回合执行 ======
async def run_episode(var_num: int, client, max_steps: int = 20):
    env = ScienceWorldWrapper(
        task_name="chemistry-mix-paint-secondary-color",
        var_num=var_num,
        step_limit=max_steps
    )
    agents = build_agent(client, env)
    team = Swarm(agents, termination_condition=HandoffTermination(target="user"))

    last_info = None
    msg_counter = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    task_done = False

    async for msg in team.run_stream(
        task=HandoffMessage(source="user", target="solo", content="Start the ScienceWorld task.")
    ):
        msg_counter += 1

        if hasattr(msg, "models_usage") and msg.models_usage:
            total_prompt_tokens += getattr(msg.models_usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(msg.models_usage, "completion_tokens", 0) or 0

        if msg.__class__.__name__ == "ToolCallExecutionEvent":
            for result in msg.content:
                try:
                    parsed = ast.literal_eval(result.content)
                    func_name = result.name
                    if func_name == "step":
                        obs = parsed.get("obs", "")
                        score = parsed.get("score", 0)
                        moves = len(parsed.get("actions", []))
                        task_done = parsed.get("done", False)
                        move_list = parsed.get("actions", [])
                        print(f"[StepResult] Score={score}, Moves={moves}, Done={task_done}, move_list={move_list}, obs={obs}")
                        last_info = parsed
                    elif func_name == "reset":
                        print(f"[ResetResult] Environment reset successfully.")
                except Exception as e:
                    print(f"  [ParseError] {e}, raw={result.content[:200]}")

        elif msg.__class__.__name__ == "HandoffMessage":
            print(f"[Handoff] {msg.source} -> {msg.target}")
            if msg.content:
                print(f"  Said: {msg.content}")

        elif msg.__class__.__name__ == "TextMessage":
            print(f" {msg.source} Said: {msg.content}")

        if msg.__class__.__name__ == "HandoffMessage" and msg.target == "user" and task_done:
            final_score = last_info.get("score", 0) if last_info else 0
            move_num = len(last_info.get("actions", [])) if last_info else 0
            success = final_score > 0
            print(f"[Episode {var_num}] completed, score={final_score}, moves={move_num}")
            return success, final_score, move_num, total_prompt_tokens, total_completion_tokens

    final_score = last_info["score"] if last_info else 0
    move_num = last_info.get("moves", 0) if last_info else env.env.get_num_moves()
    success = final_score > 0 and task_done
    return success, final_score, move_num, total_prompt_tokens, total_completion_tokens


# ====== Benchmark 主函数 ======
async def benchmark(num_episodes=5, csv_path="benchmark_results_single.csv"):
    client = OpenAIChatCompletionClient(model="gpt-5-mini")

    total_success, total_score, total_moves = 0, 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "score", "moves", "success", "prompt_tokens", "completion_tokens"])

        for var_num in range(num_episodes):
            print(f"\n=== Running episode {var_num} (Single Agent) ===")
            success, score, moves, prompt_t, completion_t = await run_episode(var_num, client)
            writer.writerow([var_num, score, moves, success, prompt_t, completion_t])
            f.flush()

            if success:
                total_success += 1
            total_score += score
            total_moves += moves
            total_prompt_tokens += prompt_t
            total_completion_tokens += completion_t

            print(f"Episode {var_num}: score={score}, moves={moves}, success={success}")

    print("\n===== Single Agent Benchmark Results =====")
    print(f"Success Rate: {total_success / num_episodes:.2%}")
    print(f"Average Score: {total_score / num_episodes:.2f}")
    print(f"Average Moves: {total_moves / num_episodes:.2f}")
    print(f"Prompt Tokens: {total_prompt_tokens}")
    print(f"Completion Tokens: {total_completion_tokens}")
    print(f"\n✅ Results saved to: {csv_path}")


if __name__ == "__main__":
    asyncio.run(benchmark())

