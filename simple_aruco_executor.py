
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_aruco_executor.py
A thin adapter so the LLM only needs to say WHAT to pick and WHERE to place.
We look up marker IDs from a YAML mapping, then call aruco_marker_tools.*
"""

import os
import yaml
from typing import Optional
from aruco_marker_tools import locate_marker_3d, pickup_by_marker, place_by_marker

# Search for mapping in a few common locations (edit/add as needed)
_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "config", "object_aruco_low.yaml"),
    os.path.join(os.path.dirname(__file__), "object_aruco_low.yaml"),
    "/mnt/data/object_aruco_low.yaml",
]

def _load_mapping() -> dict:
    for p in _SEARCH_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
    return {}

def _resolve_id(name: str, mapping: dict) -> Optional[int]:
    if not name:
        return None
    key = name.strip().lower()
    # direct match
    if key in mapping:
        return int(mapping[key])
    # tolerant match (e.g., hyphens / extra spaces)
    key2 = " ".join(key.replace("-", " ").split())
    if key2 in mapping:
        return int(mapping[key2])
    # try startswith for convenience (e.g., "bottom" -> "bottom middle")
    candidates = [k for k in mapping.keys() if k.startswith(key2)]
    if len(candidates) == 1:
        return int(mapping[candidates[0]])
    return None

def locate_by_name(name: str, camera: str = "head") -> str:
    """Locate an object/location by human-readable name using marker mapping."""
    mp = _load_mapping()
    mid = _resolve_id(name, mp)
    if mid is None:
        return f"Unknown name '{name}'. Please check object_aruco_low.yaml."
    loc = locate_marker_3d(mid, camera=camera)
    if "error" in loc:
        return f"Locate failed for '{name}' (id={mid}): {loc['error']}"
    return f"{name} (id={mid}) at base=({loc['x']:.3f},{loc['y']:.3f},{loc['z']:.3f})"

def pick_and_place_by_name(pickup: str, place: str, camera: str = "head") -> str:
    """
    The single entry point tool: pick up <pickup> and place at <place>.
    pickup/place are human-readable names which map to IDs via YAML.
    """
    mp = _load_mapping()
    pid = _resolve_id(pickup, mp)
    tid = _resolve_id(place, mp)
    if pid is None:
        return f"Unknown pickup '{pickup}'. Please update object_aruco_low.yaml."
    if tid is None:
        return f"Unknown place '{place}'. Please update object_aruco_low.yaml."
    pr = pickup_by_marker(pid, camera=camera)
    if isinstance(pr, str) and "failed" in pr.lower():
        return f"Pick step failed: {pr}"
    tr = place_by_marker(tid, camera=camera)
    return f"Done. Picked '{pickup}'(id={pid}) and placed at '{place}'(id={tid}).\nPick: {pr}\nPlace: {tr}"


def list_available_objects() -> str:
    """Return a human-readable list of all available object/location names from YAML."""
    mp = _load_mapping()
    if not mp:
        return "No objects configured. Please edit object_aruco_low.yaml."
    # stable order for deterministic prompts
    names = sorted(mp.keys())
    return "Available: " + ", ".join(names)
