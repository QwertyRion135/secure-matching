"""Cleartext model of Algorithms 3 and 4 against the plaintext reference (no MPC)."""
import pytest

from oblivsm.model import BOT, cond_assign, find_unmatched, naive_oblivious_gs, optimized_oblivious_gs
from oblivsm.plaintext_gs import assert_valid_stable, gale_shapley
from oblivsm.profiles import HANDWRITTEN, from_scores, random_profile, to_scores

SIZES = [1, 2, 3, 4, 5, 6, 8]


def _scores(s, r):
    return to_scores(s), to_scores(r)


def test_score_encoding():
    prefs = [[2, 0, 1], [1, 2, 0], [0, 1, 2]]
    assert to_scores(prefs)[0] == [1, 0, 2]
    assert from_scores(to_scores(prefs)) == prefs


def test_cond_assign():
    assert cond_assign(1, 10, 20) == 10
    assert cond_assign(0, 10, 20) == 20


def test_find_unmatched():
    assert find_unmatched([0, 0, 0]) == BOT
    for u in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1]):
        assert find_unmatched(u) != BOT


@pytest.mark.parametrize("case", HANDWRITTEN, ids=lambda c: c.name)
def test_handwritten_cases(case):
    ps, pr = _scores(case.suitor_prefs, case.reviewer_prefs)
    for res in (naive_oblivious_gs(ps, pr), optimized_oblivious_gs(ps, pr)):
        assert res.suitor_match == case.expected_suitor_match
        assert_valid_stable(case.suitor_prefs, case.reviewer_prefs, res.suitor_match, res.reviewer_match)


def test_zero_sweeps_leaves_everyone_unmatched():
    res = naive_oblivious_gs(*_scores(*random_profile(4, 0)), sweeps=0)
    assert res.suitor_match == [BOT] * 4 and res.reviewer_match == [BOT] * 4


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", range(25))
def test_optimized_equals_plaintext_gs(n, seed):
    s, r = random_profile(n, seed)
    res = optimized_oblivious_gs(*_scores(s, r))
    ref = gale_shapley(s, r)
    assert res.suitor_match == ref.suitor_match
    assert res.reviewer_match == ref.reviewer_match
    assert_valid_stable(s, r, res.suitor_match, res.reviewer_match)
    assert sorted(res.suitor_match) == list(range(n))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", range(25))
def test_naive_with_enough_sweeps_equals_plaintext_gs(n, seed):
    s, r = random_profile(n, seed)
    res = naive_oblivious_gs(*_scores(s, r), sweeps=n * n)
    assert res.suitor_match == gale_shapley(s, r).suitor_match
