import random

from arm.data import GraphConfig, Vocab, generate, to_text
from arm.data.canonical import EDGE, QUERY


def test_encode_layout_and_roundtrip():
    cfg = GraphConfig()
    vocab = Vocab(cfg.num_ids)
    p = generate(cfg, random.Random(0))
    tokens = vocab.encode(p)

    assert len(tokens) == 3 * len(p.edges) + 3
    assert tokens[0::3] == [EDGE] * len(p.edges) + [QUERY]
    assert all(0 <= t < vocab.size for t in tokens)

    expected = " ".join(
        [f"EDGE X{a} X{b}" for a, b in p.edges] + [f"QUERY X{p.source} X{p.target}"]
    )
    assert vocab.decode(tokens) == expected


def test_to_text_matches_readme_notation():
    p = generate(GraphConfig(), random.Random(1))
    lines = to_text(p).splitlines()
    assert lines[-1] == f"QUERY_REACHABLE(X{p.source}, X{p.target})"
    assert len(lines) == len(p.edges) + 1
