#!/usr/bin/env python3.10
import asyncio
import json
from scienceworld import ScienceWorldEnv
import ast
import os
import csv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from memory_tool import (
    store_triples, store_image_link, query_graph, list_entities,
    search_visual_memory_from_image, get_entity_by_vector_id,
    describe_image_from_path, get_next_task_id
)

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
            "goal":info.get("taskDesc")
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


# ====== 构建 Agents ======
def build_agents(client, env: ScienceWorldWrapper):
    user = UserProxyAgent(name="user")

    commander = AssistantAgent(
        name="commander",
        model_client=client,
        handoffs=["coder", "memory"],
        reflect_on_tool_use=True,
        tools=[env.reset, env.step],
        system_message="""
You are the Commander Agent.

Responsibilities:
- Use env.reset to start the work.
- When handing off, you should only ask the Coder to propose ONE valid next action 
(from the environment's valid action list).
- Hand off to coder when you need an action
- Wait for the Reviewer's approval or revision before executing any action.
- Do not execute actions coming directly from the Coder.
- You are not allow to pick or suggest action.
- Use env.step to execute action before ask for another action.
- Handoff the exact environment feedback (obs, reward, done, score, valid_actions) to the Coder. Do not summarize or filter it.
- No need to summarize the feedback.
- Always put the final goal on message
- Do not repeatly speak, directly handoff to other agent when you finished your speaking. Otherwise they cannot speak.
- Only when task finished (done=True in env.step), you must:
  1) Produce a final summary Like this:
Task completed. Here is the summary:
The final goal was .... , it start in the ... 
Plain summary:
Step 1: go to art studio → Obs/Result: "...", Reward=10, Score=10)
Step 2: look around → Obs: "...", Reward=0, Score=10
Step 3: pour wood cup containing yellow paint in wood cup containing blue paint → Obs: "...", Reward=20, Score=30
...
Step N: focus on green paint → Obs: "...", Reward=50, Score=100, Done=True

JSON summary:
{
  "episode": [
    {"step": 1, "action": "go to art studio", "obs": "...", "reward": 10, "score": 10, "done": false},
    {"step": 2, "action": "look around", "obs": "...", "reward": 0, "score": 10, "done": false},
    ...
    {"step": N, "action": "focus on green paint", "obs": "...", "reward": 50, "score": 100, "done": true}
  ]
}

Handoff to memory.

Special rule for ambiguity:
- If the environment returns an ambiguous action request with numbered options (0,1,2,...),
  you must explicitly ask the Coder to select exactly ONE number (as a string, e.g. "0" or "1").
  Do not try to resolve by yourself.


Handoff policy:

- You MUST always output a plain text explanation BEFORE calling transfer_to_xxx.
- The explanation must say:
  1. Why you want to handoff,
  2. Which agent is the target,
  3. What purpose the handoff will achieve.
- After the explanation, THEN call transfer_to_xxx.
- If you ever call transfer_to_xxx without explanation first, it will be considered an error.
- If you can still speak, means you did not handoff to other agents.
- Handoff to user will end the whole episode. Normally memory will handoff to user after storing the final summary. If it does not, you must handoff to user after you receiving the final summary from memory.


"""
    )

    coder = AssistantAgent(
        name="coder",
        model_client=client,
        handoffs=["reviewer","memory"],
        system_message="""
You are the Coder Agent.

You must choose one action (for reaching the final goal) and explain why. 
The action should be reasonable rather than just "valid".

After choosing, output a text message, then handoff to reviewer for validity check.

Special rule for uncertainty:
- If you are not certain which location or object is correct (for example, art studio vs workshop),
  you MUST first consult the Memory agent before proposing an action. 
  - Provide a short explanation of your uncertainty. 
  - Then transfer_to_memory, asking if any past episodes stored relevant information (e.g., where blue paint or mixing tools were found).
- After Memory responds, use that information to select exactly one valid action.
- Memory agent's response from past episodes, only use it as a reference, obvervation from env is always the truth.


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

Handoff policy:

- You MUST always output a plain text explanation BEFORE calling transfer_to_xxx.
- The explanation must say:
  1. Why you want to handoff,
  2. Which agent is the target,
  3. What purpose the handoff will achieve.
- After the explanation, THEN call transfer_to_xxx.
- If you ever call transfer_to_xxx without explanation first, it will be considered an error.
- If you can still speak, means you did not handoff to other agents.


"""
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=client,
        handoffs=["commander", "coder","memory"],
        system_message="""
You are the Reviewer Agent.
- If the Coder's action seems invalid due to ambiguous object names or repeated
  past mistakes, ask Memory if there are any stored object references or
  recipes before proposing a revision.
- You could always ask memory agent try to make the length of movement shorter.

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
- Use the object name exactly same with env's output, without any symbols.
- You MUST always output a plain text explanation BEFORE calling transfer_to_xxx.
- The explanation must say:
  1. Why you want to handoff,
  2. Which agent is the target,
  3. What purpose the handoff will achieve.
- After the explanation, THEN call transfer_to_xxx.
- If you ever call transfer_to_xxx without explanation first, it will be considered an error.
- focus on *final product* when finish, not container.
- If you can still speak, means you did not handoff to other agents.


"""
    )

    memory_agent = AssistantAgent(
        name="memory",
        model_client=client,
        handoffs=["commander","reviewer","coder","user"],
        reflect_on_tool_use=True,
        # ⚠️ 这里保持你之前的 MemoryAgent prompt，不做改动
        tools=[  # 🔹 掛上 memory_tool.py 里的工具
        store_triples,
        query_graph,
        list_entities,
        get_next_task_id
    ],
        system_message="""
You are the Memory Agent. 
Your responsibility is to manage long-term memory in Neo4j.

During the task:
- You may use query_graph(...) or list_entities() to answer questions from other agents. 
- Calling list_entities() to display all known entities.
- Do NOT guess or invent information.
- If no related memory is found, reply: "No relevant memory found." and handoff to the requester.
- Do NOT store anything during the task.
- Remember: Other agents cannot see your tool calls or results, so you must explicitly describe what you found when replying.
- Unless you handoff to other agents, otherwise they cannot speak.
- If you can still speak, means you did not handoff to other agents.

At task completion:
- The Commander will hand off a final JSON summary when the task is finished.
- You MUST parse this summary and call `store_triples(...)` **exactly once** to save all facts and events together. Such as location of rooms, objects, tools, ingredients, and the entire episode of actions, observations, rewards, and scores.
- Each triple must include its own `step` field:
  - Use `step="fact"` for persistent facts, such as task goal, outcome, or high-level conclusions.
  - Use an integer step index (e.g., 1, 2, 3, ...) for step-wise events in the episode.
- The `task_id` must be obtained via `get_next_task_id()` before calling `store_triples(...)`.
- Do not call `store_triples` multiple times per task; combine all triples into one list.
- The function call must contain only the tool invocation — no extra text, comments, or explanations around it.
- After storing, confirm completion with a short message like:
  "Stored 27 triples for task_id=3."

Example workflow:

JSON summary:
{
  "facts": [
    {"subject": "task:1", "relation": "goal", "object": "use chemistry to create green paint"},
    {"subject": "task:1", "relation": "status", "object": "completed"}
  ],
  "episode": [
    {"step": 1, "action": "go to art studio", "obs": "arrived in art studio", "reward": 10},
    {"step": 2, "action": "look around", "obs": "saw a jug", "reward": 0}
  ]
}

Store_triples example(One dict triples and one int task_id):

store_triples(
  triples=[
    {"subject": "task:1", "relation": "goal", "object": "use chemistry to create green paint", "step": "fact"},
    {"subject": "task:1", "relation": "status", "object": "completed", "step": "fact"},
    {"subject": "agent", "relation": "performed", "object": "go to art studio", "step": 1},
    {"subject": "agent", "relation": "observed", "object": "arrived in art studio", "step": 1},
    {"subject": "go to art studio", "relation": "reward", "object": "10", "step": 1},
    {"subject": "agent", "relation": "performed", "object": "look around", "step": 2},
    {"subject": "agent", "relation": "observed", "object": "saw a jug", "step": 2}
  ],
  task_id=1
)

After the store_triples call:
- Wait for tool response.
- Then respond briefly: “Stored all facts and steps for task_id=1.”
- Finally, handoff to the User to end the episode.

⚠️ CRITICAL RULES:
- Only call ONE function in each message.
- Do NOT print or describe JSON — always use a proper tool call.
- Use `get_next_task_id()` before storing.
- Do NOT reuse task_id values.
- Always ensure every triple has either a numeric step or "fact".
- Do NOT handoff to other agents after storing; only handoff to the User.
- If you can still speak, means you did not handoff to other agents.
"""

    )

    return [user, commander, coder, reviewer, memory_agent]

import asyncio
import ast
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
    task_done = False  # ⚠️ 新增：环境 done 标记

    async for msg in team.run_stream(
        task=HandoffMessage(source="user", target="commander", content="Start the ScienceWorld task")
    ):
        msg_counter += 1

        # 🔹 累计 token
        if hasattr(msg, "models_usage") and msg.models_usage:
            total_prompt_tokens += getattr(msg.models_usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(msg.models_usage, "completion_tokens", 0) or 0

        # 🔹 解析 ToolCall
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
                        print(f"[ResetResult] 初始环境加载成功 (score={parsed.get('info',{}).get('score',0)})")

                except Exception as e:
                    print(f"  [ParseError] {e}, raw={result.content[:200]}")
        elif msg.__class__.__name__ == "HandoffMessage":
              print(f"[Handoff] {msg.source} -> {msg.target}")
              if msg.content:
                  print(f"  Said: {msg.content}")
        elif msg.__class__.__name__ == "TextMessage":
              print(f" {msg.source} Said: {msg.content}")

        # 🔹 检查 handoff 到 user（说明流程完全结束了）
        if msg.__class__.__name__ == "HandoffMessage" and msg.target == "user" and task_done:
            final_score = last_info.get("score", 0) if last_info else 0
            move_num = len(last_info.get("actions", [])) if last_info else 0
            success = final_score > 0

            print(f"[Episode {var_num}] completed, score={final_score}, moves={move_num}")
            return success, final_score, move_num, total_prompt_tokens, total_completion_tokens

    # ✅ 兜底：如果没有 user handoff
    final_score = last_info["score"] if last_info else 0
    move_num = last_info.get("moves", 0) if last_info else env.env.get_num_moves()
    success = final_score > 0 and task_done
    return success, final_score, move_num, total_prompt_tokens, total_completion_tokens

# ====== Benchmark 主函数 ======
async def benchmark(num_episodes=5, csv_path="benchmark_results.csv"):
    client = OpenAIChatCompletionClient(model="gpt-5-mini")

    total_success, total_score, total_moves = 0, 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0

    # 初始化 CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "score", "moves", "success", "prompt_tokens", "completion_tokens"])

        for var_num in range(num_episodes):
            print(f"\n=== Running episode {var_num} ===")
            success, score, moves, prompt_t, completion_t = await run_episode(var_num, client)

            # 写入 CSV
            writer.writerow([var_num, score, moves, success, prompt_t, completion_t])
            f.flush()  # 即时写入

            # 累计
            if success:
                total_success += 1
            total_score += score
            total_moves += moves
            total_prompt_tokens += prompt_t
            total_completion_tokens += completion_t

            print(f"Episode {var_num}: score={score}, moves={moves}, success={success}")

    print("\n===== Benchmark Results =====")
    print(f"Success Rate: {total_success/num_episodes:.2%}")
    print(f"Average Score: {total_score/num_episodes:.2f}")
    print(f"Average Moves: {total_moves/num_episodes:.2f}")
    print(f"Prompt Tokens: {total_prompt_tokens}")
    print(f"Completion Tokens: {total_completion_tokens}")
    print(f"\n✅ Results saved to: {csv_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(benchmark())