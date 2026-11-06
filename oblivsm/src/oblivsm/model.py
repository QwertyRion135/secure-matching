"""Cleartext model of Algorithms 3 and 4: same logic as the MPC programs, plain ints.

Used as a fast test oracle. Names follow the paper: s_m/s_p = suitor's match/score,
r_m/r_p = reviewer's match/score, u_s = 1 if the suitor is unmatched.

BOT = -1 means "no partner". The paper's pseudocode uses 0 here, which clashes with
participant 0, so we use -1 like the authors' code.
"""
from __future__ import annotations

from dataclasses import dataclass

BOT = -1


def cond_assign(c: int, x: int, y: int) -> int:
    """Paper Algorithm 2: x if c == 1 else y."""
    return c * (x - y) + y


@dataclass
class ModelResult:
    suitor_match: list[int]
    reviewer_match: list[int]
    sweeps: int

    @property
    def complete(self) -> bool:
        return BOT not in self.suitor_match and BOT not in self.reviewer_match


class _State:
    def __init__(self, ps, pr):
        self.n = len(ps)
        self.ps, self.pr = ps, pr
        self.s_m = [BOT] * self.n
        self.s_p = [BOT] * self.n
        self.r_m = [BOT] * self.n
        self.r_p = [BOT] * self.n
        self.u_s = [1] * self.n


def _sweep(st: _State) -> None:
    """One pass over all suitors (the body shared by Algorithms 3 and 4)."""
    n = st.n
    for j in range(n):
        u_disp = BOT          # suitor displaced by j's new match
        r_prop = BOT          # reviewer j ends up with
        t_m, t_p = BOT, BOT   # that reviewer's new partner and score
        u_m = st.u_s[j]
        for k in range(n):
            c = int(st.s_p[j] < st.ps[j][k]) * int(st.r_p[k] < st.pr[k][j]) * u_m
            u_disp = cond_assign(c, st.r_m[k], u_disp)
            st.s_m[j] = cond_assign(c, k, st.s_m[j])
            st.s_p[j] = cond_assign(c, st.ps[j][k], st.s_p[j])
            st.u_s[j] = cond_assign(c, 0, st.u_s[j])
            r_prop = cond_assign(c, k, r_prop)
            t_m = cond_assign(c, j, t_m)
            t_p = cond_assign(c, st.pr[k][j], t_p)
        for z in range(n):
            c = int(z == u_disp)
            st.u_s[z] = cond_assign(c, 1, st.u_s[z])
            st.s_m[z] = cond_assign(c, BOT, st.s_m[z])
            st.s_p[z] = cond_assign(c, BOT, st.s_p[z])
        for z in range(n):
            c = int(z == r_prop)
            st.r_m[z] = cond_assign(c, t_m, st.r_m[z])
            st.r_p[z] = cond_assign(c, t_p, st.r_p[z])


def find_unmatched(u_s: list[int]) -> int:
    """Alg. 4 lines 5-11 as in the paper. Returns BOT iff no suitor is unmatched."""
    v, m = 0, BOT
    for p, u in enumerate(u_s):
        v = (1 - v) * u
        m = cond_assign(v, p, m)
    return m


def naive_oblivious_gs(ps, pr, sweeps: int | None = None) -> ModelResult:
    """Algorithm 3: a fixed number of sweeps (default n)."""
    st = _State(ps, pr)
    sweeps = st.n if sweeps is None else sweeps
    for _ in range(sweeps):
        _sweep(st)
    return ModelResult(list(st.s_m), list(st.r_m), sweeps)


def optimized_oblivious_gs(ps, pr, max_sweeps: int | None = None) -> ModelResult:
    """Algorithm 4: sweep until no suitor is unmatched (max_sweeps only guards against bugs)."""
    st = _State(ps, pr)
    limit = st.n * st.n + 1 if max_sweeps is None else max_sweeps
    sweeps = 0
    while find_unmatched(st.u_s) != BOT:
        if sweeps >= limit:
            raise RuntimeError("exceeded the safety bound")
        _sweep(st)
        sweeps += 1
    return ModelResult(list(st.s_m), list(st.r_m), sweeps)
