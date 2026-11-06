"""Tests on the real MP-SPDZ programs. Skipped unless MP_SPDZ_ROOT is set.

    MP_SPDZ_ROOT=/path/to/MP-SPDZ pytest -m secure -v
"""
import pytest

from oblivsm.model import naive_oblivious_gs
from oblivsm.mpspdz import run_matching, run_primitive_checks
from oblivsm.partial import (
    PARTIAL_HANDWRITTEN, assert_valid_stable_partial, gale_shapley_partial, pad_to_square,
    profile_to_scores, random_partial_profile, unpad_matching,
)
from oblivsm.plaintext_gs import assert_valid_stable, gale_shapley
from oblivsm.profiles import HANDWRITTEN, random_profile, to_scores

pytestmark = pytest.mark.secure


@pytest.fixture(scope="session")
def cache():
    return set()


def test_primitives():
    got = run_primitive_checks()
    expected = {
        "lt_true": 1, "lt_false": 0, "lt_equal": 0, "lt_bot_below_0": 1,
        "cond_select_x": 10, "cond_select_y": 20, "cond_public_x": 7,
        "eq_true": 1, "eq_false": 0, "lookup_0": 7, "lookup_2": 9,
        "unmatched_none": -1,
    }
    for name, value in expected.items():
        assert got[name] == value, name
    for name in ("unmatched_only_0", "unmatched_only_1", "unmatched_only_2", "unmatched_all"):
        assert got[name] != -1, name


@pytest.mark.parametrize("case", HANDWRITTEN, ids=lambda c: c.name)
@pytest.mark.parametrize("key", ["alg3", "alg4"])
def test_handwritten_cases(key, case, cache):
    run = run_matching(key, case.suitor_prefs, case.reviewer_prefs, compile_cache=cache)
    assert run.suitor_match == case.expected_suitor_match
    assert_valid_stable(case.suitor_prefs, case.reviewer_prefs, run.suitor_match, run.reviewer_match)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n", [3, 4, 5])
def test_alg4_equals_plaintext_reference(n, seed, cache):
    s, r = random_profile(n, seed)
    run = run_matching("alg4", s, r, compile_cache=cache)
    ref = gale_shapley(s, r)
    assert run.suitor_match == ref.suitor_match
    assert run.reviewer_match == ref.reviewer_match
    assert sorted(run.suitor_match) == list(range(n))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n", [3, 4, 5])
def test_alg3_equals_model(n, seed, cache):
    s, r = random_profile(n, seed)
    run = run_matching("alg3", s, r, compile_cache=cache)
    model = naive_oblivious_gs(to_scores(s), to_scores(r))
    assert (run.suitor_match, run.reviewer_match) == (model.suitor_match, model.reviewer_match)
    if model.complete:
        assert run.suitor_match == gale_shapley(s, r).suitor_match


# ---- partial lists ----
@pytest.mark.parametrize("case", PARTIAL_HANDWRITTEN, ids=lambda c: c.name)
@pytest.mark.parametrize("key", ["alg3_partial", "alg4_partial"])
def test_partial_handwritten_cases(key, case, cache):
    run = run_matching(key, case.suitor_prefs, case.reviewer_prefs, compile_cache=cache)
    assert run.suitor_match == case.expected_suitor_match
    assert_valid_stable_partial(case.suitor_prefs, case.reviewer_prefs, run.suitor_match, run.reviewer_match)


@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n", [3, 4])
def test_alg4_partial_equals_reference(n, seed, density, cache):
    s, r = random_partial_profile(n, seed, density)
    run = run_matching("alg4_partial", s, r, compile_cache=cache)
    ref = gale_shapley_partial(s, r)
    assert (run.suitor_match, run.reviewer_match) == (ref.suitor_match, ref.reviewer_match)


@pytest.mark.parametrize("density", [0.5, 1.0])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n", [3, 4])
def test_alg3_partial_equals_model(n, seed, density, cache):
    s, r = random_partial_profile(n, seed, density)
    run = run_matching("alg3_partial", s, r, compile_cache=cache)
    model = naive_oblivious_gs(*profile_to_scores(s, r))
    assert (run.suitor_match, run.reviewer_match) == (model.suitor_match, model.reviewer_match)


def test_unbalanced_market_via_padding(cache):
    s, r = random_partial_profile(4, 1, 0.8, n_reviewers=2)
    sp, rp, n_s, n_r = pad_to_square(s, r)
    run = run_matching("alg4_partial", sp, rp, compile_cache=cache)
    result = unpad_matching(run.suitor_match, run.reviewer_match, n_s, n_r)
    ref = gale_shapley_partial(s, r)
    assert result == (ref.suitor_match, ref.reviewer_match)
