import faiss
import numpy as np
import json
import os
from typing import List, Dict

class PersistentVectorStore:
    def __init__(self, dim: int, index_file="vector.index", json_file="vectors.json"):
        self.dim = dim
        self.index_file = index_file
        self.json_file = json_file
        self.index = faiss.IndexFlatIP(dim)  # 内积搜索
        self.id_list: List[str] = []
        self.id_to_vector: Dict[str, np.ndarray] = {}
        self.counter = 0
        self._load()

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """对向量进行归一化以便使用余弦相似度"""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _generate_id(self) -> str:
        return f"vec{self.counter:04d}"

    def add(self, vector: List[float]) -> str:
        vec_np = np.array(vector, dtype=np.float32).reshape(1, -1)
        vec_np = self._normalize(vec_np)  # 归一化
        self.index.add(vec_np)

        vec_id = self._generate_id()
        self.counter += 1

        self.id_list.append(vec_id)
        self.id_to_vector[vec_id] = vec_np

        self._save()
        return vec_id  # 返回自动分配的 ID

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        query_np = self._normalize(query_np)  # ✅ 归一化
        D, I = self.index.search(query_np, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.id_list):
                results.append({
                    "id": self.id_list[idx],
                    "similarity": float(score)  # 内积值，此时就是余弦相似度
                })
        return results

    def get_vector(self, vec_id: str) -> List[float]:
        vec = self.id_to_vector.get(vec_id)
        return vec.flatten().tolist() if vec is not None else None

    def _save(self):
        faiss.write_index(self.index, self.index_file)
        data = {
            "id_list": self.id_list,
            "vectors": {k: v.flatten().tolist() for k, v in self.id_to_vector.items()},
            "counter": self.counter
        }
        with open(self.json_file, "w") as f:
            json.dump(data, f)

    def _load(self):
        if os.path.exists(self.index_file) and os.path.exists(self.json_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.json_file, "r") as f:
                data = json.load(f)
            self.id_list = data["id_list"]
            self.id_to_vector = {k: np.array(v, dtype=np.float32) for k, v in data["vectors"].items()}
            self.counter = data.get("counter", len(self.id_list))


