#!/usr/bin/env python3.10
import os
import json
import uuid
import base64
import shutil
from datetime import datetime
from typing import List, Dict, Optional

# —— 配置 —— #
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "88888888")

# 默认图片目录与“最新图像”的假定路径（和你原系统保持一致）
DEFAULT_IMAGE_DIR = os.getenv("MEMORY_IMAGE_DIR", "/home/haoqi/Desktop/Swarmproject/image")
DEFAULT_LATEST_IMAGE = os.getenv("MEMORY_LATEST_IMAGE", f"{DEFAULT_IMAGE_DIR}/latest.jpg")

# 向量维度（open_clip ViT-L/14 OpenAI 权重输出 768d）
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "768"))

# —— 依赖 —— #
import torch
from PIL import Image
import open_clip
from neo4j import GraphDatabase

# 你项目里的持久化向量库
from vector_store import PersistentVectorStore

# OpenAI 用于（可选）图像描述
from openai import OpenAI
_openai_client = None  # 惰性初始化


# ========= 基础设施：Neo4j / VectorStore / CLIP ========= #

# 单例：Neo4j 驱动
_driver = None
def _neo4j():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver

# 单例：向量库
_vector_store = None
def _vs():
    global _vector_store
    if _vector_store is None:
        _vector_store = PersistentVectorStore(dim=VECTOR_DIM)
    return _vector_store

# 单例：CLIP（惰性加载，避免 import 即加载耗时）
_device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model = None
_clip_preprocess = None

def _ensure_clip_loaded():
    global _clip_model, _clip_preprocess
    if _clip_model is None or _clip_preprocess is None:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        _clip_model = model.eval().to(_device)
        _clip_preprocess = preprocess

def _image_to_embedding(image_path: str) -> List[float]:
    _ensure_clip_loaded()
    img = _clip_preprocess(Image.open(image_path)).unsqueeze(0).to(_device)
    with torch.no_grad():
        emb = _clip_model.encode_image(img)
    return emb[0].detach().cpu().numpy().tolist()

def _unique_image_copy(orig_path: str, base_dir: str = DEFAULT_IMAGE_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    ext = os.path.splitext(orig_path)[1] or ".jpg"
    new_path = os.path.join(base_dir, f"img_{uuid.uuid4().hex}{ext}")
    shutil.copyfile(orig_path, new_path)
    return new_path


# ================== 对外工具函数（给 Autogen 调） ================== #

def list_entities() -> str:
    """
    列出 Neo4j 中所有实体名称（升序）。
    """
    with _neo4j().session() as s:
        result = s.run("MATCH (e:Entity) RETURN DISTINCT e.name AS name ORDER BY name")
        names = [r["name"] for r in result]
    return "Known entities:\n" + "\n".join(f"- {n}" for n in names)

def get_next_task_id():
    with _neo4j().session() as s:
        result = s.run(
            """
            MATCH ()-[r]->()
            WHERE r.task_id IS NOT NULL
            RETURN coalesce(max(r.task_id), 0) + 1 AS next_id
            """
        )
        return {"task_id": result.single()["next_id"]}


def store_triples(triples: List[Dict[str, str]], task_id: int, step: Optional[int] = None) -> str:
    """
    将若干 (subject, relation, object) 三元组存入 Neo4j。
    - subject/object 转为小写
    - relation 转为大写并替换非法字符
    - 必须传入 task_id（由 agent 决定）
    - step 可选：有则作为事件(event)，无则作为事实(fact)
    """
    def _sanitize_rel(r: str) -> str:
        return (
            r.strip()
             .upper()
             .replace(" ", "_")
             .replace("-", "_")
             .replace("/", "_")
        )

    with _neo4j().session() as s:
        for t in triples:
            s_name = t["subject"].lower()
            o_name = t["object"].lower()
            rel = _sanitize_rel(t["relation"])

            query = f"""
                MERGE (a:Entity {{name: $s}})
                MERGE (b:Entity {{name: $o}})
                MERGE (a)-[r:{rel}]->(b)
                SET r.task_id = $task_id
            """
            params = {"s": s_name, "o": o_name, "task_id": task_id}

            if step is not None:
                query += ", r.step = $step"
                params["step"] = step

            s.run(query, params)

    if step is not None:
        return f"Stored {len(triples)} event triples for task_id={task_id}, step={step}."
    else:
        return f"Stored {len(triples)} fact triples for task_id={task_id}."

def query_graph(cypher: str) -> str:
    """
    执行任意 Cypher 并返回 JSON 字符串（用于调试/查询）。
    """
    with _neo4j().session() as s:
        res = s.run(cypher)
        return json.dumps([rec.data() for rec in res], indent=2)


def store_image_link(subject: str, image_path: Optional[str] = None) -> str:
    """
    存一条“视觉记忆”：
    - 将图像复制到唯一文件名
    - 生成向量，写入向量库，返回 embedding_id
    - 在 Neo4j 中建立： (subject:Entity)-[:LOOKS_LIKE]->(Image{path,time})-[:EMBEDDING]->(Vector{id})
    """
    if not image_path:
        image_path = DEFAULT_LATEST_IMAGE
    if not os.path.exists(image_path):
        return f"Error: image not found at {image_path}"

    try:
        unique_path = _unique_image_copy(image_path)
        vector = _image_to_embedding(unique_path)
        emb_id = _vs().add(vector)
        with _neo4j().session() as s:
            s.run(
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
                    "path": unique_path,
                    "time": datetime.now().isoformat(),
                    "vec_id": emb_id,
                },
            )
        return f"Stored image for {subject} as {emb_id} @ {unique_path}"
    except Exception as e:
        return f"Error in store_image_link: {e}"


def search_visual_memory_from_image(image_path: Optional[str] = None, k: int = 5) -> str:
    """
    用给定图像（默认 latest.jpg）去向量库里做 Top-k 相似搜索，返回 JSON：
    [
      {"id": "<vector_id>", "score": <float>}, ...
    ]
    """
    if not image_path:
        image_path = DEFAULT_LATEST_IMAGE
    if not os.path.exists(image_path):
        return json.dumps({"error": f"image not found at {image_path}"})
    try:
        vec = _image_to_embedding(image_path)
        results = _vs().search(vec, k)
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": f"search failed: {e}"})


def get_entity_by_vector_id(vector_id: str) -> str:
    """
    通过向量 id 反查对应的实体（如果存在 LOOKS_LIKE → EMBEDDING 关系）
    返回 JSON 列表，例如：["alice", "cup"]
    """
    with _neo4j().session() as s:
        res = s.run(
            """
            MATCH (e:Entity)-[:LOOKS_LIKE]->(:Image)-[:EMBEDDING]->(v:Vector {id: $id})
            RETURN DISTINCT e.name AS name
            """,
            {"id": vector_id},
        )
        names = [r["name"] for r in res]
        return json.dumps(names, indent=2)


def describe_image_from_path(image_path: Optional[str] = None,
                             task_instruction: Optional[str] = None,
                             model: str = "gpt-4o",
                             detail: str = "high",
                             temperature: float = 0.2) -> str:
    """
    （可选）用 OpenAI 多模态描述本地图像（离线场景可不使用）
    """
    path = image_path or DEFAULT_LATEST_IMAGE
    if not os.path.exists(path):
        return f"Error: image not found at {path}"

    instruction = task_instruction or "Describe in plain text what is visible in the image."

    # 懒加载 OpenAI 客户端
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": detail}},
                ],
            }],
        )
        # 适配 text 返回
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error in describe_image_from_path: {e}"


# 可选：资源回收
def _close():
    """关闭底层资源（如果需要时调用）。"""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
