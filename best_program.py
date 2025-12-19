from __future__ import annotations
"""
initial_program.py

EVOLVE TARGET:
    plan_parallel_flip_sequence(points, edges_start, edges_goal, max_steps, seed)

Returns:
    List[List[Edge]]  # steps, each is list of edges to flip in parallel
"""

from typing import List, Set
import random
from triangulation_ops import Triangulation, canon_edge, Edge, Point


def plan_parallel_flip_sequence(
    points: List[Point],
    edges_start: Set[Edge],
    edges_goal: Set[Edge],
    max_steps: int = 50,
    seed: int = 0,
) -> List[List[Edge]]:
    rng = random.Random(seed)
    T = Triangulation.from_edges(points, edges_start)
    if T.edges == edges_goal:
        return []

    # EVOLVE-BLOCK-START
    seq: List[List[Edge]] = []
    for _ in range(max_steps):
        if T.edges == edges_goal:
            break

        extra = T.edges - edges_goal
        missing = edges_goal - T.edges

        candidates = random.sample(list(extra), min(len(extra), 12))  # Allow up to 12 candidates for more flexibility

        chosen: List[Edge] = []
        used_tri = set()

        for e in candidates:
            if len(chosen) < 8 and (sup := T.flippable_support(e)) is not None:
                u, v, a, b = sup
                new_e = canon_edge(a, b)
                gain = 1 if new_e in missing else 0
                t1, t2 = T.incident_triangles_of_edge(u, v, a, b)
                if t1 not in used_tri and t2 not in used_tri:
                    if gain == 0 and rng.random() < 0.70:  # Further adjust acceptance probability
                        continue
                    used_tri.update([t1, t2])
                    chosen.append(e)

        if not chosen:
            # fallback: single random flippable edge
            chosen = [e for e in random.sample(list(T.edges), min(400, len(T.edges))) if T.flippable_support(e)]  # Simplified fallback mechanism
            if chosen:
                chosen = [chosen[0]]  # Take the first valid candidate

        if not chosen:
            break

        if not T.apply_parallel_flip_step(chosen):
            break
        seq.append(chosen)

    return seq
    # EVOLVE-BLOCK-END
