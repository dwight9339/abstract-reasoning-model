"""Procedural generator for directed-graph reachability problems.

Each problem is a directed graph plus a query (source, target). The label is
whether ``target`` can be reached from ``source`` by following edges forward.

Design choices (these matter more than they look):

1. Exact difficulty control. Every problem is built around a chain
   ``source -> v1 -> ... -> v_depth = target``, so we know exactly how many
   hops the answer depends on.

2. Minimal-pair negatives via a degree-preserving edge swap. Every graph gets
   the chain plus one extra edge ``x -> w`` between two distractor nodes. A
   positive keeps ``v_k -> v_{k+1}`` and ``x -> w``; a negative swaps their
   heads to ``v_k -> w`` and ``x -> v_{k+1}``, cutting the chain at a uniformly
   random point. Every node keeps exactly the same in- and out-degree and the
   edge count is identical, so the only way to tell positives from negatives is
   to follow the wiring. (An earlier version reversed a chain edge instead; that
   leaked the answer through the degrees of the source and target whenever the
   first or last edge was the one reversed.)

3. Label-blind distractors. Extra nodes and random edges are added so the
   chain is not the whole graph. The positive and negative versions of each
   graph are built side by side, and a random edge is kept only if it leaves
   *both* answers intact (no shortcut in the positive, no bypass around the cut
   in the negative). The distractor edges are therefore chosen without knowing
   the label, so their statistics cannot give it away. (Checking only the
   version being emitted leaked the label at shallow depths: positives accepted
   almost any edge near the source and target, negatives did not.)

4. Names carry no meaning. Node IDs are a random sample from a fixed pool and
   the edge list is shuffled, so neither an ID nor a position in the input says
   anything about the node's role.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphConfig:
    num_ids: int = 64  # size of the node-ID pool; must cover the largest graph
    min_depth: int = 1
    max_depth: int = 8
    min_distractors: int = 4  # extra nodes on top of the depth + 1 chain nodes (>= 2)
    max_distractors: int = 12
    extra_edges_per_node: float = 1.0  # random edges *attempted*, relative to node count
    p_positive: float = 0.5


@dataclass(frozen=True)
class Problem:
    edges: tuple[tuple[int, int], ...]  # directed (src, dst) pairs, using pool IDs
    source: int
    target: int
    label: bool  # True if target is reachable from source
    depth: int  # length of the source-target chain (hops needed for a positive)
    num_nodes: int  # nodes allocated, including any distractors left without edges


def shortest_distances(edges, source: int) -> dict[int, int]:
    """BFS from ``source``; returns {node: hop count} for every reachable node."""
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nxt in adj.get(node, ()):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def generate(cfg: GraphConfig, rng: random.Random) -> Problem:
    if cfg.min_distractors < 2:
        raise ValueError("min_distractors must be >= 2 (the edge swap needs two distractors)")
    depth = rng.randint(cfg.min_depth, cfg.max_depth)
    label = rng.random() < cfg.p_positive
    n = depth + 1 + rng.randint(cfg.min_distractors, cfg.max_distractors)
    if n > cfg.num_ids:
        raise ValueError(f"graph needs {n} nodes but the ID pool has only {cfg.num_ids}")

    # Work in local indices first: 0..depth is the chain, the rest are distractors.
    source, target = 0, depth
    cut = rng.randrange(depth)
    x, w = rng.sample(range(depth + 1, n), 2)
    chain = {(i, i + 1) for i in range(depth) if i != cut}
    positive = chain | {(cut, cut + 1), (x, w)}
    negative = chain | {(cut, w), (x, cut + 1)}

    extras: set[tuple[int, int]] = set()
    for _ in range(round(cfg.extra_edges_per_node * n)):
        edge = (rng.randrange(n), rng.randrange(n))
        if edge[0] == edge[1] or edge in positive or edge in negative or edge in extras:
            continue
        extras.add(edge)
        pos_ok = shortest_distances(positive | extras, source).get(target) == depth
        neg_ok = target not in shortest_distances(negative | extras, source)
        if not (pos_ok and neg_ok):
            extras.remove(edge)

    edges = (positive if label else negative) | extras

    # Relabel with random pool IDs and shuffle so nothing leaks through names or order.
    ids = rng.sample(range(cfg.num_ids), n)
    edge_list = [(ids[a], ids[b]) for a, b in sorted(edges)]
    rng.shuffle(edge_list)
    return Problem(
        edges=tuple(edge_list),
        source=ids[source],
        target=ids[target],
        label=label,
        depth=depth,
        num_nodes=n,
    )


def generate_many(cfg: GraphConfig, count: int, seed: int) -> list[Problem]:
    rng = random.Random(seed)
    return [generate(cfg, rng) for _ in range(count)]
