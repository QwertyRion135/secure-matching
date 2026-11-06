"""Preference profiles: score encoding, random inputs and hand-written test cases.

Lists are ordered best -> worst. The oblivious algorithms use scores instead
(higher = better: favourite = n-1, last choice = 0, -1 = "no partner").
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

Prefs = list[list[int]]


def to_scores(prefs: Prefs) -> list[list[int]]:
    n = len(prefs)
    scores = [[0] * n for _ in prefs]
    for i, lst in enumerate(prefs):
        for pos, x in enumerate(lst):
            scores[i][x] = n - 1 - pos
    return scores


def from_scores(scores: list[list[int]]) -> Prefs:
    return [sorted(range(len(scores)), key=lambda x: -row[x]) for row in scores]


def random_profile(n: int, seed: int) -> tuple[Prefs, Prefs]:
    rng = random.Random(seed)

    def one_side() -> Prefs:
        rows = []
        for _ in range(n):
            lst = list(range(n))
            rng.shuffle(lst)
            rows.append(lst)
        return rows

    return one_side(), one_side()


def benchmark_profile(n: int, seed: int) -> tuple[Prefs, Prefs]:
    """Fixed input for (n, seed), the same on every machine."""
    return random_profile(n, seed * 1_000_003 + n)


@dataclass(frozen=True)
class Case:
    name: str
    suitor_prefs: Prefs
    reviewer_prefs: Prefs
    expected_suitor_match: Optional[list[int]]  # the suitor-optimal stable matching


def immediate_match_case(n: int = 3) -> Case:
    """Everyone's first choice is mutual and distinct."""
    lists = [[(i + d) % n for d in range(n)] for i in range(n)]
    return Case(f"immediate_match_{n}", lists, [list(r) for r in lists], list(range(n)))


def long_sequence_case(n: int) -> Case:
    """Many proposals: all suitors rank 0..n-1, all reviewers rank n-1..0. Suitor i gets n-1-i."""
    suitors = [list(range(n)) for _ in range(n)]
    reviewers = [list(range(n - 1, -1, -1)) for _ in range(n)]
    return Case(f"long_sequence_{n}", suitors, reviewers, [n - 1 - i for i in range(n)])


HANDWRITTEN: list[Case] = [
    Case("single_pair", [[0]], [[0]], [0]),
    immediate_match_case(3),
    # two stable matchings exist; the suitor-optimal one is [0, 1]
    Case("two_by_two_suitor_optimal", [[0, 1], [1, 0]], [[1, 0], [0, 1]], [0, 1]),
    Case("cyclic_three",
         [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
         [[1, 2, 0], [2, 0, 1], [0, 1, 2]],
         [0, 1, 2]),
    # s0 is bumped by s1, s1 by s2, then they settle for r1 / r2
    Case("rejection_chain_three",
         [[0, 1, 2], [0, 2, 1], [0, 1, 2]],
         [[2, 1, 0], [0, 1, 2], [0, 1, 2]],
         [1, 2, 0]),
    long_sequence_case(4),
]
