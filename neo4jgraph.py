#!/usr/bin/env python3.10
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ========= 配置 =========
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "88888888"
MAX_LABEL_LEN = 8
LIMIT = 150
SAVE_PATH_PNG = os.path.expanduser("~/Desktop/neo4j_graph_named.png")
SAVE_PATH_SVG = os.path.expanduser("~/Desktop/neo4j_graph_named.svg")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def truncate(text, max_len=MAX_LABEL_LEN):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."

def fetch_graph(tx):
    """从 Neo4j 拉取节点与关系"""
    result = tx.run(f"MATCH (a)-[r]->(b) RETURN a, type(r) AS rel_type, r, b LIMIT {LIMIT}")
    G = nx.DiGraph()

    for record in result:
        a, r, b = record["a"], record["r"], record["b"]
        rel_type = record["rel_type"] or "RELATION"

        # 节点名称
        a_name = a.get("name") or a.get("id") or next(iter(a.labels), "Node")
        b_name = b.get("name") or b.get("id") or next(iter(b.labels), "Node")

        # 关系类型识别
        rel_name = rel_type
        if hasattr(r, "_properties") and r._properties:
            props = r._properties
            if "type" in props:
                rel_name = props["type"].lower()
            elif "relation" in props:
                rel_name = props["relation"].lower()

        G.add_node(a.element_id, label=truncate(a_name))
        G.add_node(b.element_id, label=truncate(b_name))
        G.add_edge(a.element_id, b.element_id, rel=rel_name)

    return G

def infer_relation_color(rel_name: str) -> str:
    """根据关系语义上色"""
    text = rel_name.lower()
    if "goal" in text:
        return "darkgreen"
    elif "perform" in text:
        return "blue"
    elif "reward" in text:
        return "purple"
    elif "status" in text or "complet" in text or "done" in text:
        return "red"
    elif "observ" in text or "focus" in text or "see" in text:
        return "orange"
    return "gray"

# ========= 执行 =========
with driver.session() as session:
    G = session.execute_read(fetch_graph)

plt.figure(figsize=(20, 14))

# ✅ 调整布局：更强的分离度、更高迭代次数
pos = nx.spring_layout(G, k=5, iterations=500, seed=42)

# 节点样式
nx.draw_networkx_nodes(G, pos, node_color="#AEDFF7", node_size=1100, alpha=0.9, edgecolors="black")

# 边颜色
edge_colors = [infer_relation_color(G.edges[e]["rel"]) for e in G.edges()]
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, alpha=0.6, width=1.4)

# 节点文字
nx.draw_networkx_labels(
    G, pos,
    labels={n: G.nodes[n]['label'] for n in G.nodes()},
    font_size=8, font_color="black",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
)

# ✅ 图例
legend_patches = [
    mpatches.Patch(color="darkgreen", label="goal"),
    mpatches.Patch(color="blue", label="performed"),
    mpatches.Patch(color="purple", label="reward"),
    mpatches.Patch(color="red", label="status/completed"),
    mpatches.Patch(color="orange", label="observed/focus"),
    mpatches.Patch(color="gray", label="other"),
]
plt.legend(
    handles=legend_patches,
    loc="upper right",
    fontsize=9,
    frameon=True,
    title="Relation Type",
    title_fontsize=10,
    facecolor="white",
    edgecolor="black",
)

plt.title("Neo4j Memory Graph (Colored by Relation Type)", fontsize=14, fontweight="bold")
plt.axis("off")

# 导出高清图像
plt.savefig(SAVE_PATH_PNG, dpi=600, bbox_inches="tight")
plt.savefig(SAVE_PATH_SVG, format="svg", bbox_inches="tight")
print(f"✅ Graph saved to:\n  PNG: {SAVE_PATH_PNG}\n  SVG: {SAVE_PATH_SVG}")

