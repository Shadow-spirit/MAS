#!/usr/bin/env python3.10
import asyncio
import json
import csv
import os
import re
import ast
import random
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from memory_tool import (
    store_triples, query_graph, list_entities, get_next_task_id
)

# ============= 环境封装部分 =============
import alfworld
import alfworld.agents.environment

import yaml
import alfworld
import alfworld.agents.environment

class AlfworldWrapper:
    def __init__(self, gamefile, step_limit=10):
        print("Initializing AlfredTWEnv with uploaded config...")
        
        with open("alfworld_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # 强制覆盖必要字段
        config["split"] = "eval_out_of_distribution"
        config["env"]["max_episode_length"] = step_limit

        self.main_env = getattr(
            alfworld.agents.environment, config["env"]["type"]
        )(config, train_eval=config["split"])

        self.main_env.game_files = [gamefile]
        self.env = self.main_env.init_env(batch_size=1)
        self.env.reset()

        self.episode_actions = []
        print(f"✅ AlfworldWrapper 初始化完成 -> {gamefile}")

    def reset(self):
        self.done = False
        obs, info = self.env.reset()
        self.episode_actions = []

        obs_text = obs[0] if isinstance(obs, list) else str(obs)
        goal = info.get("goal", ["unknown goal"])[0] if isinstance(info.get("goal"), list) else info.get("goal", "")
        valid_actions = info.get("valid", [["look around"]])[0]


        return {
            "obs": obs_text,
            "actions": [],
            "done": False
        }
        

    def step(self, action: str):
        observation, reward, done, info = self.env.step([action])
        self.episode_actions.append(action)

        return {
            "obs": observation,
            "reward": float(reward[0]),
            "done": bool(done[0]),
            "score": info.get("score", [0])[0] if "score" in info else 0,
            "actions": list(self.episode_actions),
        }


    # ===== 辅助函数 =====
    @staticmethod
    def _process_ob(ob):
        if ob.startswith("You arrive at loc "):
            ob = ob[ob.find(". ") + 2 :]
        return ob

    @staticmethod
    def _process_action(action: str):
        action = action.strip().replace("<", "").replace(">", "")
        action = action.replace("OK.", "").replace("OK", "").strip()
        return action


# ============= 构建 Agents 部分 =============
def build_agents(client, env: AlfworldWrapper):
    user = UserProxyAgent(name="user")

    commander = AssistantAgent(
        name="commander",
        model_client=client,
        handoffs=["coder", "memory"],
        reflect_on_tool_use=True,
        tools=[env.reset, env.step],
        system_message="""You are the Commander Agent for ALFWorld tasks.
- calling env.reset() to initialize.
- calling env.step(action) to execute.Action list means all the actions you have done in this episode. 
- You must use env.step(action) run the action proposed by reviewer before ask coder for next action.
- Coordinate with coder and reviewer.
- Handoff the environment feedback to coder directly. They won't see tool calls or results. Especially the obs information.
- Always put the task goal in handoff message.
- You should only ask coder for one action at a time. The reviewer will verify it and handoff you the result.
- You are not allow to make action by yourself, when you need an action, just handoff the enviroment feed back to Coder, Reviewer will handoff you an action later.
- When task finished summarize the task and handoff to memory.
- You are NOT allow to suggest any action or provide example to coder.
- If you see you are repeating speaking, call handoff function to coder directly.
"""
    )

    coder = AssistantAgent(
        name="coder",
        model_client=client,
        handoffs=["reviewer", "memory"],
        system_message="""
You are the Coder Agent. Your job is to propose ONE valid next action in ALFWorld.

You are now in a household environment called Alfworld, and your tasks include locating objects, heating or cooling items, and other similar activities.

Special rule for uncertainty:
- If you are not certain which location or object is correct,
  you MUST first consult the Memory agent before proposing an action. 
  - Provide a short explanation of your uncertainty. 
  - Then transfer_to_memory, asking if any past episodes stored relevant information (e.g., where blue paint or mixing tools were found).
- After Memory responds, use that information to select exactly one valid action.
- Memory agent's response from past episodes, only use it as a reference, obvervation from env is always the truth.
- Commander will handover the environment obs in first handoff message, you must use it to help you decide the next action.
NOTE:
- You can only interact with the object in front of you, go to the receptacle first if you want to interact with it.
- Below are the allowed command templates. You must pick one action from follow these EXACTLY:
  look:                             look around your current location
  inventory:                        check your current inventory
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):  place an object in or on a receptacle
  examine (something):              examine a receptacle or an object
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object


- You must check carefully whether your output command is consistent with the allowed commands above!!! Any output that is not among the commands listed above is not permitted!!!
- Always output your reasoning first, then your chosen action.
- After outputting your command, hand off to the Reviewer for verification.
- Please only choose one action at a time.
- If you can still speak, means you did not handoff to other agents.
- You can only handoff to one agent at a time. No silent handoff.


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
        handoffs=["commander", "coder", "memory"],
        system_message="""
You are the Reviewer Agent. Your responsibility is to verify whether the Coder's proposed action in ALFWorld strictly follows syntactic and logical rules.

You are now in a household environment called Alfworld, and your tasks include locating objects, heating or cooling items, and other similar activities.

NOTE:
- You can only interact with the object in front of you, go to the receptacle first if you want to interact with it.
- You must check the Coder's action strictly follows these templates : 
  look:                             look around your current location
  inventory:                        check your current inventory
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):  place an object in or on a receptacle
  examine (something):              examine a receptacle or an object
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object


If the Coder's action does not conform exactly to one of these patterns, reject it and revise it.
If valid, respond with 'approved: <action>' and hand off to the Commander.
Handoff to commander only when you approved the action.
If invalid, correct it with 'revised: <new_action>' and hand off to the commander
You must provide an action to the commander. No silent handoff.


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

    memory_agent = AssistantAgent(
        name="memory",
        model_client=client,
        handoffs=["commander","reviewer","coder","user"],
        reflect_on_tool_use=True,
        tools=[store_triples, query_graph, list_entities, get_next_task_id],
        system_message="""You are the Memory Agent. 
Your responsibility is to manage long-term memory in Neo4j.

During the task:
- You may use query_graph(...) or list_entities() to answer questions from other agents. 
- Calling list_entities() to display all known entities.
- Do NOT guess or invent information.
- If no related memory is found, reply: "No relevant memory found." and handoff to the requester.
- Do NOT store anything during the task.
- Do not give any recommend action.
- Remember: Other agents cannot see your tool calls or results, so you must explicitly describe what you found when replying.
- Unless you handoff to other agents, otherwise they cannot speak.
- If you can still speak, means you did not handoff to other agents.

At task completion:
- The Commander will hand off a final summary when the task is finished.
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
- Wait for confirmation that the triples were stored successfully.
- ONLY when you want finish the whole episode, you can handoff to user otherwise handoff to the requester.
"""
    )

    return [user, commander, coder, reviewer, memory_agent]



# ============= 单轮任务执行 =============
async def run_episode(task, client, max_steps=20):
    gamefile = task["gamefile"]
    goal = task["goal"]

    print(f"\n=== Running: {goal} ===")
    env = AlfworldWrapper(gamefile, step_limit=max_steps)
    agents = build_agents(client, env)
    team = Swarm(agents, termination_condition=HandoffTermination(target="user"))

    last_info = None
    msg_counter = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    task_done = False  # ⚠️ 新增：环境 done 标记

    async for msg in team.run_stream(
        task=HandoffMessage(source="user", target="commander", content="Start the ALFworld task")
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
                  print(msg)
        else:
              print(f"  {msg.content}")

        # 🔹 检查 handoff 到 user（说明流程完全结束了）
        if msg.__class__.__name__ == "HandoffMessage" and msg.target == "user" and task_done:
            final_score = last_info.get("score", 0) if last_info else 0
            move_num = len(last_info.get("actions", [])) if last_info else 0
            success = last_info.get("done", False)

            print(f"[Episode] completed, score={final_score}, moves={move_num}")
            return success, final_score, move_num, total_prompt_tokens, total_completion_tokens

    # ✅ 兜底：如果没有 user handoff
    final_score = last_info["score"] if last_info else 0
    move_num = last_info.get("moves", 0) if last_info else env.env.get_num_moves()
    success = final_score > 0 and task_done
    return success, final_score, move_num, total_prompt_tokens, total_completion_tokens

# ============= Benchmark 主函数 =============
async def benchmark_from_json(subset_path="soapbar_subset.json", csv_path="benchmark_results_ALF.csv", max_steps=20):
    client = OpenAIChatCompletionClient(model="gpt-5-mini")#,reasoning = {"effort":"medium"},text = {"verbosity":"low"}

    with open(subset_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(["episode", "task_goal", "score", "moves", "success", "prompt_tokens", "completion_tokens"])

        for i, task in enumerate(tasks):
            if i > 1:
                success, score, moves, prompt_t, completion_t = await run_episode(task, client)
                writer.writerow([i, task["goal"], score, moves, success, prompt_t, completion_t])
                f_csv.flush()

    print("\n✅ Benchmark finished. Results saved to:", csv_path)


if __name__ == "__main__":
    asyncio.run(benchmark_from_json())

