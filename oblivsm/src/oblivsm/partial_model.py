"""Cleartext model of the oblivious algorithms on partial lists (extension).

Algorithm 3 is unchanged (use naive_oblivious_gs with partial scores). Algorithm 4's
"stop when nobody is unmatched" never ends if someone can't be matched, so we stop when
a sweep makes no progress: the sum of the reviewers' scores only grows, and grows
whenever a proposal is accepted, so an unchanged sum means nothing more can happen.
"""
from __future__ import annotations

from .model import ModelResult, _State, _sweep
from .partial import pad_to_square, profile_to_scores, unpad_matching


def optimized_partial(ps, pr, max_sweeps: int | None = None) -> ModelResult:
    st = _State(ps, pr)
    limit = st.n * st.n + 2 if max_sweeps is None else max_sweeps
    sweeps = 0
    while True:
        if sweeps >= limit:
            raise RuntimeError("exceeded the safety bound")
        total_before = sum(st.r_p)
        _sweep(st)
        sweeps += 1
        if sum(st.r_p) == total_before:
            break
    return ModelResult(list(st.s_m), list(st.r_m), sweeps)


def solve_rectangular_model(suitor_prefs, reviewer_prefs):
    """Pad to a square market, run the model, drop the dummies. Returns (s_m, r_m)."""
    sp, rp, n_s, n_r = pad_to_square(suitor_prefs, reviewer_prefs)
    res = optimized_partial(*profile_to_scores(sp, rp))
    return unpad_matching(res.suitor_match, res.reviewer_match, n_s, n_r)
