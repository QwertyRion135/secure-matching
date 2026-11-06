"""Plaintext Gale-Shapley and a stability checker (the reference for all tests).

Suitors and reviewers are numbered 0..n-1. Preference lists are ordered best -> worst
and must be permutations of 0..n-1. suitor_match[s] is the reviewer of suitor s,
reviewer_match[r] the suitor of reviewer r (-1 = unmatched).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterator, Optional, Sequence

Prefs = Sequence[Sequence[int]]


@dataclass(frozen=True)
class GSResult:
    suitor_match: list[int]
    reviewer_match: list[int]
    proposals: int


def validate_preferences(suitor_prefs: Prefs, reviewer_prefs: Prefs) -> list[str]:
    """List of problems with the input (empty = valid)."""
    n = len(suitor_prefs)
    if len(reviewer_prefs) != n:
        return [f"market must be balanced: {n} suitors but {len(reviewer_prefs)} reviewers"]
    errors = []
    for side, prefs in (("suitor", suitor_prefs), ("reviewer", reviewer_prefs)):
        for i, lst in enumerate(prefs):
            if sorted(lst) != list(range(n)):
                errors.append(f"{side} {i}: list {list(lst)} is not a permutation of 0..{n - 1}")
    return errors


def _rank_table(prefs: Prefs) -> list[list[int]]:
    """rank[i][x] = position of x in i's list (0 = best)."""
    rank = [[0] * len(prefs) for _ in prefs]
    for i, lst in enumerate(prefs):
        for pos, x in enumerate(lst):
            rank[i][x] = pos
    return rank


def gale_shapley(suitor_prefs: Prefs, reviewer_prefs: Prefs) -> GSResult:
    """Suitor-proposing deferred acceptance (gives the suitor-optimal stable matching)."""
    problems = validate_preferences(suitor_prefs, reviewer_prefs)
    if problems:
        raise ValueError("invalid preference profile: " + "; ".join(problems))

    n = len(suitor_prefs)
    reviewer_rank = _rank_table(reviewer_prefs)
    next_choice = [0] * n
    holder = [-1] * n
    free = list(range(n - 1, -1, -1))
    proposals = 0

    while free:
        s = free.pop()
        r = suitor_prefs[s][next_choice[s]]
        next_choice[s] += 1
        proposals += 1
        current = holder[r]
        if current == -1:
            holder[r] = s
        elif reviewer_rank[r][s] < reviewer_rank[r][current]:
            holder[r] = s
            free.append(current)
        else:
            free.append(s)

    suitor_match = [-1] * n
    for r, s in enumerate(holder):
        suitor_match[s] = r
    return GSResult(suitor_match, list(holder), proposals)


def check_matching(suitor_prefs: Prefs, reviewer_prefs: Prefs, suitor_match: Sequence[int],
                   reviewer_match: Optional[Sequence[int]] = None) -> list[str]:
    """Violations of 'valid, perfect, stable matching' (empty list = all good)."""
    errors = validate_preferences(suitor_prefs, reviewer_prefs)
    if errors:
        return errors
    n = len(suitor_prefs)
    if len(suitor_match) != n:
        return [f"suitor_match has length {len(suitor_match)}, expected {n}"]

    broken = False
    owner: dict[int, int] = {}
    for s, r in enumerate(suitor_match):
        if r == -1:
            errors.append(f"suitor {s} is unmatched")
        elif not 0 <= r < n:
            errors.append(f"suitor {s} matched to out-of-range reviewer {r}")
            broken = True
        elif r in owner:
            errors.append(f"reviewer {r} is matched to more than one suitor ({owner[r]} and {s})")
            broken = True
        else:
            owner[r] = s

    if reviewer_match is not None:
        if len(reviewer_match) != n:
            errors.append(f"reviewer_match has length {len(reviewer_match)}, expected {n}")
            broken = True
        else:
            for s, r in enumerate(suitor_match):
                if 0 <= r < n and reviewer_match[r] != s:
                    errors.append(f"inconsistent: suitor {s} -> reviewer {r} but "
                                  f"reviewer {r} -> suitor {reviewer_match[r]}")
            for r, s in enumerate(reviewer_match):
                if s != -1 and not (0 <= s < n and suitor_match[s] == r):
                    errors.append(f"inconsistent: reviewer {r} -> suitor {s} but suitor "
                                  f"{s} is matched elsewhere")
    if broken:
        return errors

    suitor_rank, reviewer_rank = _rank_table(suitor_prefs), _rank_table(reviewer_prefs)
    partner_of_reviewer = [-1] * n
    for s, r in enumerate(suitor_match):
        if r != -1:
            partner_of_reviewer[r] = s
    for s in range(n):
        cur_r = suitor_match[s]
        for r in range(n):
            if r == cur_r:
                continue
            s_prefers = cur_r == -1 or suitor_rank[s][r] < suitor_rank[s][cur_r]
            cur_s = partner_of_reviewer[r]
            r_prefers = cur_s == -1 or reviewer_rank[r][s] < reviewer_rank[r][cur_s]
            if s_prefers and r_prefers:
                errors.append(f"blocking pair: suitor {s} and reviewer {r}")
    return errors


def assert_valid_stable(suitor_prefs, reviewer_prefs, suitor_match, reviewer_match=None):
    errors = check_matching(suitor_prefs, reviewer_prefs, suitor_match, reviewer_match)
    if errors:
        raise AssertionError("; ".join(errors))


def enumerate_stable_matchings(suitor_prefs: Prefs, reviewer_prefs: Prefs) -> Iterator[list[int]]:
    """Every stable perfect matching, by brute force (n <= 8)."""
    if len(suitor_prefs) > 8:
        raise ValueError("brute force is only meant for n <= 8")
    for perm in permutations(range(len(suitor_prefs))):
        if not check_matching(suitor_prefs, reviewer_prefs, list(perm)):
            yield list(perm)


def is_suitor_optimal(suitor_prefs: Prefs, reviewer_prefs: Prefs, suitor_match: Sequence[int]) -> bool:
    """True iff every suitor does at least as well as in any other stable matching."""
    rank = _rank_table(suitor_prefs)
    for other in enumerate_stable_matchings(suitor_prefs, reviewer_prefs):
        if any(rank[s][suitor_match[s]] > rank[s][other[s]] for s in range(len(suitor_prefs))):
            return False
    return True
