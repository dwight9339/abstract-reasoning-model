"""Canonical token representation of a reachability problem.

This is the abstract format the reasoning core is trained on:

    EDGE X12 X3  EDGE X3 X40  ...  QUERY X12 X40

Each node ID gets its own token. The answer (reachable or not) is not part of
the sequence; it is the training target.
"""

from __future__ import annotations

from .graphs import Problem

PAD, EDGE, QUERY = 0, 1, 2
_SPECIAL_NAMES = ["<pad>", "EDGE", "QUERY"]


class Vocab:
    def __init__(self, num_ids: int):
        self.num_ids = num_ids

    @property
    def size(self) -> int:
        return len(_SPECIAL_NAMES) + self.num_ids

    def node_token(self, node_id: int) -> int:
        if not 0 <= node_id < self.num_ids:
            raise ValueError(f"node id {node_id} outside pool of {self.num_ids}")
        return len(_SPECIAL_NAMES) + node_id

    def encode(self, problem: Problem) -> list[int]:
        tokens: list[int] = []
        for a, b in problem.edges:
            tokens += [EDGE, self.node_token(a), self.node_token(b)]
        tokens += [QUERY, self.node_token(problem.source), self.node_token(problem.target)]
        return tokens

    def token_name(self, token: int) -> str:
        if token < len(_SPECIAL_NAMES):
            return _SPECIAL_NAMES[token]
        return f"X{token - len(_SPECIAL_NAMES)}"

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self.token_name(t) for t in tokens if t != PAD)


def to_text(problem: Problem) -> str:
    """Readable form, matching the notation in the README."""
    lines = [f"EDGE(X{a}, X{b})" for a, b in problem.edges]
    lines.append(f"QUERY_REACHABLE(X{problem.source}, X{problem.target})")
    return "\n".join(lines)
