from collections import Counter

import pytest

from arm.data import GraphConfig, generate_many

CFG = GraphConfig(min_depth=1, max_depth=12, min_distractors=2, max_distractors=16)


@pytest.fixture(scope="module")
def problems():
    return generate_many(CFG, count=3000, seed=0)


def reachable_set(edges, source):
    """Independent oracle (deliberately not reusing the generator's BFS)."""
    seen, stack = {source}, [source]
    while stack:
        node = stack.pop()
        for a, b in edges:
            if a == node and b not in seen:
                seen.add(b)
                stack.append(b)
    return seen


def shortest_hops(edges, source, target):
    frontier, seen, hops = {source}, {source}, 0
    while frontier:
        if target in frontier:
            return hops
        frontier = {b for a, b in edges if a in frontier} - seen
        seen |= frontier
        hops += 1
    return None


def test_deterministic_for_a_seed():
    assert generate_many(CFG, 50, seed=7) == generate_many(CFG, 50, seed=7)
    assert generate_many(CFG, 50, seed=7) != generate_many(CFG, 50, seed=8)


def test_labels_match_oracle(problems):
    for p in problems:
        assert (p.target in reachable_set(p.edges, p.source)) == p.label


def test_positive_depth_is_exact_shortest_path(problems):
    for p in problems:
        if p.label:
            assert shortest_hops(p.edges, p.source, p.target) == p.depth


def test_well_formed(problems):
    for p in problems:
        assert len(set(p.edges)) == len(p.edges)
        assert all(a != b for a, b in p.edges)
        assert all(0 <= x < CFG.num_ids for e in p.edges for x in e)
        assert CFG.min_depth <= p.depth <= CFG.max_depth


def test_labels_balanced_at_every_depth(problems):
    by_depth = Counter((p.depth, p.label) for p in problems)
    for d in range(CFG.min_depth, CFG.max_depth + 1):
        pos, neg = by_depth[(d, True)], by_depth[(d, False)]
        assert 0.35 < pos / (pos + neg) < 0.65, (d, pos, neg)


def test_ids_are_not_informative(problems):
    # The source should not always get the same ID (or a small set of them).
    assert len({p.source for p in problems}) > CFG.num_ids // 2


def best_threshold_accuracy(values, labels):
    """Accuracy of the best rule of the form `value >= t` (or its negation)."""
    best = 0.0
    for t in set(values):
        acc = sum((v >= t) == y for v, y in zip(values, labels)) / len(labels)
        best = max(best, acc, 1 - acc)
    return best


@pytest.fixture(scope="module")
def shallow_problems():
    # Shallow problems are where leaks are hardest to avoid, so check them separately.
    return generate_many(GraphConfig(min_depth=1, max_depth=2, min_distractors=2), 4000, seed=1)


@pytest.mark.parametrize("subset", ["problems", "shallow_problems"])
@pytest.mark.parametrize(
    "feature",
    [
        "target_in_degree",
        "target_out_degree",
        "source_in_degree",
        "source_out_degree",
        "num_edges",
        "num_source_nodes",  # nodes with in-degree 0
        "num_sink_nodes",  # nodes with out-degree 0
    ],
)
def test_no_simple_shortcut_predicts_label(request, subset, feature):
    """If a one-number heuristic can guess the answer, a network will learn that
    heuristic instead of learning to reason. Keep every such cue near chance."""

    def compute(p):
        ins = Counter(b for _, b in p.edges)
        outs = Counter(a for a, _ in p.edges)
        nodes = {x for e in p.edges for x in e}
        return {
            "target_in_degree": ins[p.target],
            "target_out_degree": outs[p.target],
            "source_in_degree": ins[p.source],
            "source_out_degree": outs[p.source],
            "num_edges": len(p.edges),
            "num_source_nodes": sum(ins[x] == 0 for x in nodes),
            "num_sink_nodes": sum(outs[x] == 0 for x in nodes),
        }[feature]

    problems = request.getfixturevalue(subset)
    values = [compute(p) for p in problems]
    labels = [p.label for p in problems]
    assert best_threshold_accuracy(values, labels) < 0.55


def test_rejects_too_few_distractors():
    with pytest.raises(ValueError):
        generate_many(GraphConfig(min_distractors=1), 1, seed=0)
