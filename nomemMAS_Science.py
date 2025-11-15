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


# ====== 构建 Agents (No-Memory baseline, 保留 handoff policy) ======
def build_agents(client, env: ScienceWorldWrapper):
    user = UserProxyAgent(name="user")

    commander = AssistantAgent(
        name="commander",
        model_client=client,
        handoffs=["coder","user"],  # ❌ 无 memory
        reflect_on_tool_use=True,
        tools=[env.reset, env.step],
        system_message="""
You are the Commander Agent.

Responsibilities:
- Use env.reset to start the ScienceWorld task.
- After reset, always ask the Coder to choose one valid next action.
- Use env.step to execute the chosen action and send the feedback back to the Coder.
- Never invent or guess actions.
- When done=True in env.step, summarize all steps and the outcome in JSON format and handoff to the user to end this task, otherwise it will not end.
- Handoff the exact environment feedback (obs, reward, done, score, valid_actions) to the Coder. Do not summarize or filter it.
- No need to summarize the feedback.
- If moves >=20, also handoff to user directly to end this task.
- Do not keep talking, handoff to other agents to continue the task.

Handoff policy:
- You MUST always output a plain text explanation BEFORE calling transfer_to_xxx.
- The explanation must say:
  1. Why you want to handoff,
  2. Which agent is the target,
  3. What purpose the handoff will achieve.
- After the explanation, THEN call transfer_to_xxx.
- If you ever call transfer_to_xxx without explanation first, it will be considered an error.
- If you can still speak, it means you did not actually handoff.
"""
    )

    coder = AssistantAgent(
        name="coder",
        model_client=client,
        handoffs=["reviewer"],
        system_message="""
You are the Coder Agent.

Responsibilities:
- You must choose one action (for reaching the final goal) and explain why. 
- The action should be reasonable rather than just "valid".
- Explain your reasoning briefly.
- After proposing an action, handoff to the Reviewer for approval.

CRITICAL RULES:
- You must ONLY select an action string that matches EXACTLY one of the templates in the current valid_actions list.
- Replace "OBJ" in the template with the exact object names shown in the observation.But remove brackets
- Do not invent verbs or action formats.
- Do not repeat actions.
- mix OBJ mean mix two OBJ in same container. pour OBJ into OBJ mean pour one OBJ into another OBJ.
- If object contain something, use this format, "<action> <container> containing <OBJ>". No brackets.
- For multiple containing "<action> <container> containing <OBJ1> and <OBJ2>". Link with "and". Also no brackets.
- look in action allow you to check what in container.
Important : When the final goal is achieved, you must focus on the final product itself.Do NOT focus on the container.(e.g focus on wood cup containing orange juice -> focus on orange juice)
- When you need an object name, use look around to observe.

Handoff policy:
- You MUST always output a plain text explanation BEFORE calling transfer_to_xxx.
- The explanation must say:
  1. Why you want to handoff,
  2. Which agent is the target,
  3. What purpose the handoff will achieve.
- After the explanation, THEN call transfer_to_xxx.
- If you can still speak, it means you did not actually handoff.
"""
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=client,
        handoffs=["commander", "coder"],
        system_message="""
You are the Reviewer Agent.

Responsibilities:
- Check the Coder's proposed action for both validity and usefulness.
- If valid: reply with 'approved: <action>' and handoff to the Commander.
- If invalid: reply with 'revised: <new_action>' and handoff back to the Coder.

Responsibilities:
- Check both validity AND usefulness of the Coder's proposed action.
- Do not approve actions that do not advance toward the final goal.
- If valid and useful: respond with `approved: <action>` and hand off to commander.
- If invalid/useless: explain briefly why, then propose a corrected version using `revised: <new_action>` back to coder.
- Never just handoff without giving feedback.
- Only handoff if the task/episode is finished, or after you approve/revise an action.
- Only one action once rather than a plan.
- If there are better solutions, reject and propose revision.
- mix OBJ mean mix two OBJ in same container. pour OBJ into OBJ mean pour one OBJ into another OBJ. Those two actions are for liquid only.
- If object contain something, use this format, "<action> OBJ containing OBJ". No brackets.
- For multiple containing "<action> <container name> containing a and b". Link with "and".
- If the coder proposes to focus on the container holding the product,reject it and revise to "focus on <final product>".
Important : When the final goal is achieved, you must focus on the final product itself.Do NOT focus on the container.(e.g focus on wood cup containing orange juice -> focus on orange juice)
- Do not keep repeating the same action.
- No known action matches that input. Could because both no valid action or no such OBJ.


Handoff policy:
- You MUST always output a plain text explanation BEFORE calling transfer_to_xxx.
- The explanation must say:
  1. Why you want to handoff,
  2. Which agent is the target,
  3. What purpose the handoff will achieve.
- After the explanation, THEN call transfer_to_xxx.
- If you can still speak, it means you did not actually handoff.
"""
    )

    return [user, commander, coder, reviewer]


# ====== 单回合执行 ======
async def run_episode(var_num: int, client, max_steps: int = 20):
    env = ScienceWorldWrapper(
        task_name="chemistry-mix-paint-secondary-color",
        var_num=var_num,
        step_limit=max_steps
    )
    agents = build_agents(client, env)
    team = Swarm(agents, termination_condition=HandoffTermination(target="user"))

    last_info = None
    msg_counter = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    task_done = False

    async for msg in team.run_stream(
        task=HandoffMessage(source="user", target="commander", content="Start the ScienceWorld task")
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
async def benchmark(num_episodes=3, csv_path="benchmark_results_nomemory.csv"):
    client = OpenAIChatCompletionClient(model="gpt-5-mini")

    total_success, total_score, total_moves = 0, 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "score", "moves", "success", "prompt_tokens", "completion_tokens"])

        for var_num in range(num_episodes):
            print(f"\n=== Running episode {var_num} (No-Memory) ===")
            success, score, moves, prompt_t, completion_t = await run_episode(var_num+2, client)

            writer.writerow([var_num, score, moves, success, prompt_t, completion_t])
            f.flush()

            if success:
                total_success += 1
            total_score += score
            total_moves += moves
            total_prompt_tokens += prompt_t
            total_completion_tokens += completion_t

            print(f"Episode {var_num}: score={score}, moves={moves}, success={success},prompt_token={prompt_t}, completion_token={completion_t}")

    print("\n===== No-Memory Benchmark Results =====")
    print(f"Success Rate: {total_success / num_episodes:.2%}")
    print(f"Average Score: {total_score / num_episodes:.2f}")
    print(f"Average Moves: {total_moves / num_episodes:.2f}")
    print(f"Prompt Tokens: {total_prompt_tokens}")
    print(f"Completion Tokens: {total_completion_tokens}")
    print(f"\n✅ Results saved to: {csv_path}")


if __name__ == "__main__":
    asyncio.run(benchmark())

