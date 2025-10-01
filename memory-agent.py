#!/usr/bin/env python3.10
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from neo4j import GraphDatabase
import openai, re, json, asyncio
from typing import List, Tuple
from typing import List, Dict# Neo4j setup
from datetime import datetime
import base64
import os
from PIL import Image
from io import BytesIO
import torch
import open_clip
from PIL import Image
from vector_store import PersistentVectorStore
from openai import OpenAI
client = OpenAI()
vector_store = PersistentVectorStore(dim=768)


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model,_, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')

def image_to_embedding(image_path: str) -> List[float]:
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image)
    return embedding[0].cpu().numpy().tolist()

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "88888888"))

def load_image_base64(image_path: str = "/images/latest.jpg") -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


import uuid, shutil

def _unique_image_path(orig_path: str, base_dir: str = "/home/haoqi/Desktop/Swarmproject/image") -> str:
    os.makedirs(base_dir, exist_ok=True)
    ext = os.path.splitext(orig_path)[1] or ".jpg"
    new_path = os.path.join(base_dir, f"img_{uuid.uuid4().hex}{ext}")
    try:
        shutil.copyfile(orig_path, new_path)  # 复制到唯一命名的文件
    except Exception as e:
        raise RuntimeError(f"copy image failed: {e}")
    return new_path

def store_image_link(subject: str, image_path: str) -> str:
    try:
        if not os.path.exists(image_path):
            return f"Error: image not found at {image_path}"
        # 每次保存都落一个唯一文件名，避免 path 冲突/覆盖
        unique_path = _unique_image_path(image_path)
        vector = image_to_embedding(unique_path)
    except Exception as e:
        return f"Error in image_to_embedding: {e}"

    try:
        embedding_id = vector_store.add(vector)
    except Exception as e:
        return f"Error in FAISS add: {e}"

    try:
        with driver.session() as session:
            session.run(
                """
                MERGE (p:Entity {name: $subject})
                MERGE (img:Image {path: $path})
                SET img.time = $time
                MERGE (vec:Vector {id: $vec_id})
                MERGE (p)-[:LOOKS_LIKE]->(img)
                MERGE (img)-[:EMBEDDING]->(vec)
                """,
                {
                    "subject": subject.lower(),
                    "path": unique_path,  # <-- 用唯一路径
                    "time": datetime.now().isoformat(),
                    "vec_id": embedding_id,
                },
            )
        return f"Stored image for {subject} as {embedding_id} @ {unique_path}"
    except Exception as e:
        return f"Error in Neo4j: {e}"

    

def store_triples(triples: List[Dict[str, str]]) -> str:
    """
    Stores triples into Neo4j. Each triple is a dict with keys: subject, relation, object.
    """
    with driver.session() as session:
        for triple in triples:
            s = triple["subject"]
            r = triple["relation"]
            o = triple["object"]
            session.run(
                f"""
                MERGE (a:Entity {{name: $s}})
                MERGE (b:Entity {{name: $o}})
                MERGE (a)-[:{r.upper()}]->(b)
                """,
                {"s": s, "o": o}
            )
    return f"Stored {len(triples)} triples."


def query_graph(cypher: str) -> str:
    """
    Execute a Cypher query and return results.
    """
    with driver.session() as session:
        result = session.run(cypher)
        return json.dumps([record.data() for record in result], indent=2)

def list_entities() -> str:
    """
    Return a list of all distinct entity names stored in Neo4j.
    """
    uri = "bolt://localhost:7687" 
    driver = GraphDatabase.driver(uri, auth=("neo4j", "88888888"))

    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN DISTINCT e.name AS name ORDER BY name")
        entities = [record["name"] for record in result]
    driver.close()

    return "Known entities:\n" + "\n".join(f"- {e}" for e in entities)

def search_visual_memory_from_image(image_path: str, k: int = 5) -> str:
    vector = image_to_embedding(image_path)
    results = vector_store.search(vector, k)
    return json.dumps(results, indent=2)

def get_entity_by_vector_id(vector_id: str) -> str:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[:LOOKS_LIKE]->(:Image)-[:EMBEDDING]->(v:Vector {id: $id})
            RETURN DISTINCT e.name AS name
            """,
            {"id": vector_id}
        )
        records = [record["name"] for record in result]
        return json.dumps(records, indent=2)

import os, base64
def describe_image_from_path(image_path: str, task_instruction: str = None) -> str:
    if not os.path.exists(image_path):
        return f"Error: image not found at {image_path}"

    if not task_instruction:
        task_instruction = "Describe in plain text what is visible in the image."

    # Encode local file as data URL
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": task_instruction},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ],
        }],
    )
    return resp.choices[0].message.content


# Autogen setup
async def main():
    client = OpenAIChatCompletionClient(model="gpt-4o")
    user = UserProxyAgent("user")

    memory_agent = AssistantAgent(
        name="memory",
        model_client=client,
        tools=[describe_image_from_path,store_triples, query_graph,list_entities,store_image_link,search_visual_memory_from_image,get_entity_by_vector_id],
        handoffs=["user"],
        system_message="""
You are a Memory Agent.

Your responsibilities:
- Always begin by calling list_entities() to display all known entities.
- Extract structured triples from user input and store them using store_triples(...)
- Remember anything that can be expressed as a triple, even if the user didn’t explicitly ask you to.
- Answer memory-related questions by generating Cypher and calling query_graph(...) and display the result.

-All memory have a propery called task_id, whenever you descovered a related operation or object, always check the whole task to provide a better result
Memory Storage Schema:
- All entities are stored in Neo4j as lowercase strings. Match entity names or properties exactly—do not match on labels.
- Use CONTAINS or similar operators if partial matching is required, but always on entity properties, not node labels.

- When creating triples:
    * Convert subject and object to lowercase strings.
    * The relation must be a valid Neo4j relationship type containing only uppercase letters, digits, and underscores.
    * Do not include spaces, hyphens, slashes, or other illegal characters in the relation.
    * If the relation contains spaces or other invalid characters, replace them with underscores. 
      For example, "is a" → "IS_A", "has gender" → "HAS_GENDER".

Visual Memory Handling:
- If the user mentions a visual input but does not provide an image path, assume the image is located at /home/haoqi/Desktop/Swarmproject/image/latest.jpg.
- To store a visual memory, call store_image_link(subject, image_path).
- To retrieve the most similar image memory, call search_visual_memory_from_image(image_path).
- If the user asks "Who is this?" or similar questions referring to an image, assume the image path is /home/haoqi/Desktop/Swarmproject/image/latest.jpg and call search_visual_memory_from_image accordingly.


IMPORTANT:
- After storing or answering, always Handoff to the user.
- This ends your turn. Do NOT continue speaking unless the user speaks again.
- You may have multiple users. Always link memory to the correct user context.

When asked about an entity, use the following Cypher query pattern:

MATCH (e:Entity {name: "<ENTITY_NAME>"})-[r]-(connected)
RETURN e, r, connected

Use relation type and direction to infer semantic meaning when appropriate.

"""
    )

    team = Swarm([user, memory_agent], termination_condition=HandoffTermination(target="user"))

    while True:
        task = input("💬 Ask or record something ('exit' to quit): ")
        if task.strip().lower() in ["exit", "quit"]:
            break
        await Console(team.run_stream(task=HandoffMessage(source="user", target="memory", content=task)))

if __name__ == "__main__":
    asyncio.run(main())
