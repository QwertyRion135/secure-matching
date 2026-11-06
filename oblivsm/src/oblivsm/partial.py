"""Partial preference lists (extension).

A participant lists only the partners they accept. A pair can be matched only if both
list each other, and anyone may stay unmatched (-1). The two sides may differ in size:
suitor lists hold reviewer indices and vice versa.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

Prefs = Sequence[Sequence[int]]


def validate_partial(suitor_prefs: Prefs, reviewer_prefs: Prefs) -> list[str]:
    errors = []
    n_s, n_r = len(suitor_prefs), len(reviewer_prefs)
    for side, prefs, bound in (("suitor", suitor_prefs, n_r), ("reviewer", reviewer_prefs, n_s)):
        for i, lst in enumerate(prefs):
            if len(set(lst)) != len(lst):
                errors.append(f"{side} {i}: list {list(lst)} lists a partner twice")
            for x in lst:
                if not 0 <= x < bound:
                    errors.append(f"{side} {i}: entry {x} is out of range 0..{bound - 1}")
    return errors


def _rank_maps(prefs: Prefs) -> list[dict[int, int]]:
    return [{x: pos for pos, x in enumerate(lst)} for lst in prefs]


@dataclass(frozen=True)
class PartialGSResult:
    suitor_match: list[int]
    reviewer_match: list[int]
    proposals: int


def gale_shapley_partial(suitor_prefs: Prefs, reviewer_prefs: Prefs) -> PartialGSResult:
    """Deferred acceptance with incomplete lists (suitor-optimal result)."""
    problems = validate_partial(suitor_prefs, reviewer_prefs)
    if problems:
        raise ValueError("invalid preference profile: " + "; ".join(problems))
    n_s, n_r = len(suitor_prefs), len(reviewer_prefs)
    reviewer_rank = _rank_maps(reviewer_prefs)
    next_choice = [0] * n_s
    holder = [-1] * n_r
    free = list(range(n_s - 1, -1, -1))
    proposals = 0

    while free:
        s = free.pop()
        if next_choice[s] >= len(suitor_prefs[s]):
            continue                      # out of reviewers: stays unmatched
        r = suitor_prefs[s][next_choice[s]]
        next_choice[s] += 1
        proposals += 1
        if s not in reviewer_rank[r]:
            free.append(s)                # r did not list s
            continue
        current = holder[r]
        if current == -1:
            holder[r] = s
        elif reviewer_rank[r][s] < reviewer_rank[r][current]:
            holder[r] = s
            free.append(current)
        else:
            free.append(s)

    suitor_match = [-1] * n_s
    for r, s in enumerate(holder):
        if s != -1:
            suitor_match[s] = r
    return PartialGSResult(suitor_match, list(holder), proposals)


def check_matching_partial(suitor_prefs: Prefs, reviewer_prefs: Prefs, suitor_match: Sequence[int],
                           reviewer_match: Optional[Sequence[int]] = None) -> list[str]:
    """Violations of 'valid and stable' (empty = ok). Unmatched people are allowed."""
    errors = validate_partial(suitor_prefs, reviewer_prefs)
    if errors:
        return errors
    n_s, n_r = len(suitor_prefs), len(reviewer_prefs)
    if len(suitor_match) != n_s:
        return [f"suitor_match has length {len(suitor_match)}, expected {n_s}"]

    s_rank, r_rank = _rank_maps(suitor_prefs), _rank_maps(reviewer_prefs)
    broken = False
    owner: dict[int, int] = {}
    for s, r in enumerate(suitor_match):
        if r == -1:
            continue
        if not 0 <= r < n_r:
            errors.append(f"suitor {s} matched to out-of-range reviewer {r}")
            broken = True
            continue
        if r not in s_rank[s]:
            errors.append(f"suitor {s} is matched to reviewer {r}, whom he did not list")
            broken = True
        if s not in r_rank[r]:
            errors.append(f"reviewer {r} is matched to suitor {s}, whom she did not list")
            broken = True
        if r in owner:
            errors.append(f"reviewer {r} is matched to more than one suitor ({owner[r]} and {s})")
            broken = True
        owner.setdefault(r, s)

    if reviewer_match is not None:
        if len(reviewer_match) != n_r:
            errors.append(f"reviewer_match has length {len(reviewer_match)}, expected {n_r}")
            broken = True
        else:
            for s, r in enumerate(suitor_match):
                if 0 <= r < n_r and reviewer_match[r] != s:
                    errors.append(f"inconsistent: suitor {s} -> reviewer {r} but "
                                  f"reviewer {r} -> suitor {reviewer_match[r]}")
            for r, s in enumerate(reviewer_match):
                if s != -1 and not (0 <= s < n_s and suitor_match[s] == r):
                    errors.append(f"inconsistent: reviewer {r} -> suitor {s} but suitor "
                                  f"{s} is matched elsewhere")
    if broken:
        return errors

    partner_of_reviewer = [-1] * n_r
    for s, r in enumerate(suitor_match):
        if r != -1:
            partner_of_reviewer[r] = s
    for s in range(n_s):
        cur_r = suitor_match[s]
        for r in suitor_prefs[s]:
            if r == cur_r or s not in r_rank[r]:
                continue
            s_prefers = cur_r == -1 or s_rank[s][r] < s_rank[s][cur_r]
            cur_s = partner_of_reviewer[r]
            r_prefers = cur_s == -1 or r_rank[r][s] < r_rank[r][cur_s]
            if s_prefers and r_prefers:
                errors.append(f"blocking pair: suitor {s} and reviewer {r}")
    return errors


def assert_valid_stable_partial(suitor_prefs, reviewer_prefs, suitor_match, reviewer_match=None):
    errors = check_matching_partial(suitor_prefs, reviewer_prefs, suitor_match, reviewer_match)
    if errors:
        raise AssertionError("; ".join(errors))


def enumerate_stable_matchings_partial(suitor_prefs: Prefs, reviewer_prefs: Prefs) -> Iterator[list[int]]:
    """Every stable matching, by brute force (sizes <= 6)."""
    n_s, n_r = len(suitor_prefs), len(reviewer_prefs)
    if max(n_s, n_r) > 6:
        raise ValueError("brute force is only meant for sizes <= 6")
    current: list[int] = []
    used: set[int] = set()

    def rec(s: int) -> Iterator[list[int]]:
        if s == n_s:
            if not check_matching_partial(suitor_prefs, reviewer_prefs, current):
                yield list(current)
            return
        for r in [-1, *suitor_prefs[s]]:
            if r != -1 and r in used:
                continue
            current.append(r)
            if r != -1:
                used.add(r)
            yield from rec(s + 1)
            current.pop()
            if r != -1:
                used.discard(r)

    yield from rec(0)


def is_suitor_optimal_partial(suitor_prefs: Prefs, reviewer_prefs: Prefs, suitor_match: Sequence[int]) -> bool:
    def rank(s, r):  # unmatched is worse than any listed reviewer
        return len(suitor_prefs[s]) if r == -1 else list(suitor_prefs[s]).index(r)

    for other in enumerate_stable_matchings_partial(suitor_prefs, reviewer_prefs):
        if any(rank(s, suitor_match[s]) > rank(s, other[s]) for s in range(len(suitor_prefs))):
            return False
    return True


def random_partial_profile(n_suitors: int, seed: int, density: float,
                           n_reviewers: Optional[int] = None) -> tuple[list[list[int]], list[list[int]]]:
    """Each participant lists each possible partner with probability `density`, in random order."""
    n_r = n_suitors if n_reviewers is None else n_reviewers
    rng = random.Random(seed)

    def side(count: int, others: int) -> list[list[int]]:
        rows = []
        for _ in range(count):
            lst = [x for x in range(others) if rng.random() < density]
            rng.shuffle(lst)
            rows.append(lst)
        return rows

    return side(n_suitors, n_r), side(n_r, n_suitors)


@dataclass(frozen=True)
class PartialCase:
    name: str
    suitor_prefs: list[list[int]]
    reviewer_prefs: list[list[int]]
    expected_suitor_match: list[int]


PARTIAL_HANDWRITTEN: list[PartialCase] = [
    PartialCase("single_empty", [[]], [[]], [-1]),
    PartialCase("empty_lists", [[], []], [[], []], [-1, -1]),
    # suitor 0 lists reviewer 0 but she lists nobody, so they cannot match
    PartialCase("one_sided_listing", [[0], [1]], [[], [1]], [-1, 1]),
    # s1 beats s0 at r1; s0 has nobody else; s2 lists nobody; r0 is left single
    PartialCase("subset_with_exhausted_suitor", [[1], [1, 0], []], [[1], [1, 0], []], [-1, 1, -1]),
    # s1 bumps s0 from r0, s0 falls back to r1, s2 is rejected by r0 and takes r2
    PartialCase("rejection_chain_partial", [[0, 1], [0], [0, 2]], [[1, 0, 2], [0], [2]], [1, 0, 2]),
]


# Scores: favourite of a list of length L = L-1, last choice = 0, unlisted = -1.
def to_scores_partial(prefs: Prefs, n_other: int) -> list[list[int]]:
    rows = []
    for lst in prefs:
        row = [-1] * n_other
        for pos, x in enumerate(lst):
            row[x] = len(lst) - 1 - pos
        rows.append(row)
    return rows


def profile_to_scores(suitor_prefs: Prefs, reviewer_prefs: Prefs):
    return (to_scores_partial(suitor_prefs, len(reviewer_prefs)),
            to_scores_partial(reviewer_prefs, len(suitor_prefs)))


def to_input_rows_partial(prefs: Prefs, n_other: int) -> list[list[int]]:
    """Scores + 1, so the MP-SPDZ input files hold no negative numbers."""
    return [[v + 1 for v in row] for row in to_scores_partial(prefs, n_other)]


# Unbalanced markets: pad the smaller side with dummies who list nobody and are listed by nobody.
def pad_to_square(suitor_prefs: Prefs, reviewer_prefs: Prefs):
    n_s, n_r = len(suitor_prefs), len(reviewer_prefs)
    size = max(n_s, n_r)
    sp = [list(lst) for lst in suitor_prefs] + [[] for _ in range(size - n_s)]
    rp = [list(lst) for lst in reviewer_prefs] + [[] for _ in range(size - n_r)]
    return sp, rp, n_s, n_r


def unpad_matching(suitor_match: Sequence[int], reviewer_match: Sequence[int], n_s: int, n_r: int):
    s_m, r_m = list(suitor_match[:n_s]), list(reviewer_match[:n_r])
    if any(r >= n_r for r in s_m) or any(s >= n_s for s in r_m):
        raise ValueError("a real participant was matched to a dummy")
    return s_m, r_m
