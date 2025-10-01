#!/usr/bin/env python3.10
import asyncio
import json
from scienceworld import ScienceWorldEnv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from memory_tool import (
    store_triples, store_image_link, query_graph, list_entities,
    search_visual_memory_from_image, get_entity_by_vector_id, describe_image_from_path, get_next_task_id
)
# ====== 环境封装 ======
class ScienceWorldWrapper:
    def __init__(self, task_name="chemistry-mix-paint-secondary-color", var_num=0, step_limit=50):
        self.env = ScienceWorldEnv("", None, envStepLimit=step_limit)
        self.env.load(task_name, var_num,
                      simplificationStr="teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction",
                      generateGoldPath=True)
        self.episode_actions = []
        self.task_name = task_name

    def reset(self):
        obs, info = self.env.reset()
        self.episode_actions = []
        return {"obs": obs, "info": info}

    def step(self, action: str):
        obs, reward, done, info = self.env.step(action)
        self.episode_actions.append(action)
        return {
            "obs": obs,
            "reward": reward,
            "done": done,
            "valid_actions": info["valid"],
            "actions": list(self.episode_actions)
        }
    def get_score(self):
        """在 episode 结束时调用，获取最终总分"""
        return None

# ====== 构建 Agents ======
def build_agents(client, env: ScienceWorldWrapper):
    user = UserProxyAgent(name="user")

    commander = AssistantAgent(
        name="commander",
        model_client=client,
        handoffs=["coder", "memory","user"],
        reflect_on_tool_use=True,
        tools=[env.reset, env.step,env.get_score],
        system_message="""
You are the Commander Agent.

Responsibilities:
- Use env.reset to start the work.
- When handing off, you should only ask the Coder to propose ONE valid next action 
(from the environment's valid action list).
- Hand off to coder when you need an action
- Wait for the Reviewer’s approval or revision before executing any action.
- Do not execute actions coming directly from the Coder.
- You are not allow to pick or suggest action.
- Use env.step to execute action before ask for another action
- Summary the obs information before handoff.
- Always put the final goal on message
- Do not repeatly speak, directly handoff to other agent when you finished your speaking.
- Do not update obs summary by your own, unless env proposed.
- If object contain something, use this format, "<action> OBJ contaning OBJ".
- For multiple containing "<action> OBJ contaning OBJ and OBJ". Link with "and".
- When task finished (done=True in env.step), you must:
  1) Produce a final summary (obs, actions, reward, done=True).
  2) Handoff this summary to Memory agent for storage.
  3) After Memory confirms, handoff to User for exit.
- When task finished, use get_score for final score
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


CRITICAL RULES:
- You must ONLY select an action string that matches EXACTLY one of the templates in the current valid_actions list.
- Replace "OBJ" in the template with the exact object names shown in the observation.
- Do not invent verbs or action formats.

- look in action allow you to check what in container.
- When the final goal is achieved (e.g., green paint created inside a container),
  do NOT focus on the container string (e.g., "wood cup now containing green paint").
  Instead, focus directly on the final product itself (e.g., "focus on green paint"),
  because the environment only accepts the product name for task completion.

- When you need an object name, use look around to observe.

Rules for choosing actions:
- You must only output actions exactly as they appear in the valid_actions list.
- If object contain something, use this format, "<action> OBJ containing OBJ".
- For multiple containing "<action> OBJ containing OBJ and OBJ". Link with "and".
- Use focus on action carefully, if focus on wrong object could cause a huge reward decrease.
-No known action matches that input. Could because both no valid action or no such OBJ.
Handoff policy:

- You MUST always output a plain text explanation BEFORE calling transfer_to_xxx.
- The explanation must say:
  1. Why you want to handoff,
  2. Which agent is the target,
  3. What purpose the handoff will achieve.
- After the explanation, THEN call transfer_to_xxx.
- If you ever call transfer_to_xxx without explanation first, it will be considered an error.


"""
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=client,
        handoffs=["commander", "coder","memory"],
        system_message="""
You are the Reviewer Agent.
- If the Coder’s action seems invalid due to ambiguous object names or repeated
  past mistakes, ask Memory if there are any stored object references or
  recipes before proposing a revision.
- You could always ask memory agent try to make the length of movement shorter.
Responsibilities:
- Check both validity AND usefulness of the Coder’s proposed action.
- Do not approve actions that do not advance toward the final goal.
- If valid and useful: respond with `approved: <action>` and hand off to commander.
- If invalid/useless: explain briefly why, then propose a corrected version using `revised: <new_action>` back to coder.
- Never just handoff without giving feedback.
- Only handoff if the task/episode is finished, or after you approve/revise an action.
- Only one action once rather than a plan.
- The action must be in valid_actions.
- If there are better solutions, reject and propose revision.
- If object contain something, use this format, "<action> OBJ containing OBJ". 
- For multiple containing "<action> OBJ containing OBJ and OBJ". Link with "and".
- If the coder proposes to focus on the container holding the product,
  reject it and revise to "focus on <final product>".
- Example: If the observation shows "wood cup now containing green paint",
  the correct action is "focus on green paint", not "focus on wood cup".

-No known action matches that input. Could because both no valid action or no such OBJ.

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


"""
    )

    memory_agent = AssistantAgent(
        name="memory",
        model_client=client,
        handoffs=["commander","reviewer","coder"],
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

Rules:
1. During a task (before completion):
   - Do NOT store triples automatically.
   - Only perform queries when explicitly requested by other agents (e.g., Commander or Coder).
   - Never perform silent handoff. Always explain the purpose of the handoff in plain text first.
   - If you need to store something mid-task, always request a task_id first (via get_next_task_id or Commander-provided id).
   - When storing mid-task, distinguish:
     • Facts (timeless knowledge, e.g., "goal → is → create green paint", "art studio" → "contain" → "table") → store_triples(..., task_id, step=None)
     • Events (time-ordered actions/observations) → store_triples(..., task_id, step=<integer step>)

2. At task completion (Commander sends a final summary with 'done=True' or explicitly says task is finished):
   - Extract structured triples (in {subject, relation, object} format) from the final summary.
   - Use the *exact* object strings from env observations or actions. No renaming, no simplification.
   - Request or use the current task_id before storing.
   - Store facts and events with store_triples(), including task_id and step values where appropriate.
   - Confirm storage with a short message, e.g., "Stored 12 triples (task_id=5)."

3. Forbidden behaviors:
   - Do NOT repeatedly store trivial facts during the task (like "hallway contains air").
   - Do NOT query the graph unless explicitly asked.
   - Do NOT perform silent handoffs.

4. Always handoff back to the agent who asked, after answering.

Examples:
- Mid-task:
  Commander: "Memory, have we stored salt water recipe before?"
  Memory: Perform query_graph, return results.

- Task completion:
  Commander: "Task completed. Summary: ... done=True"
  Memory: Extract triples → call store_triples(triples, task_id, step=...) → reply "Stored 15 triples (task_id=5)."

"""
    )

    return [user, commander, coder, reviewer, memory_agent]

# ====== 主程序 ======
async def main():
    env = ScienceWorldWrapper(task_name="chemistry-mix-paint-secondary-color")  # 默认任务: mix-paint
    client = OpenAIChatCompletionClient(model="gpt-5-mini",)
    agents = build_agents(client, env)

    team = Swarm(agents, termination_condition=HandoffTermination(target="user"))

    print("=== MAS + ScienceWorld Demo ===")
    await Console(team.run_stream(
        task=HandoffMessage(source="user", target="commander", content="Start the ScienceWorld task")
    ))

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())

