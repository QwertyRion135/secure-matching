"""Partial preference lists and unbalanced markets (no MPC)."""
import pytest

from oblivsm.model import naive_oblivious_gs, optimized_oblivious_gs
from oblivsm.partial import (
    PARTIAL_HANDWRITTEN, assert_valid_stable_partial, check_matching_partial,
    enumerate_stable_matchings_partial, gale_shapley_partial, is_suitor_optimal_partial,
    pad_to_square, profile_to_scores, random_partial_profile, to_input_rows_partial,
    to_scores_partial, unpad_matching, validate_partial,
)
from oblivsm.partial_model import optimized_partial, solve_rectangular_model
from oblivsm.plaintext_gs import gale_shapley

DENSITIES = [0.0, 0.3, 0.6, 1.0]


# ---- reference and checker ----
@pytest.mark.parametrize("case", PARTIAL_HANDWRITTEN, ids=lambda c: c.name)
def test_handwritten_expected_matching(case):
    res = gale_shapley_partial(case.suitor_prefs, case.reviewer_prefs)
    assert res.suitor_match == case.expected_suitor_match
    assert_valid_stable_partial(case.suitor_prefs, case.reviewer_prefs, res.suitor_match, res.reviewer_match)
    assert is_suitor_optimal_partial(case.suitor_prefs, case.reviewer_prefs, res.suitor_match)


@pytest.mark.parametrize("sp,rp,matching,reviewer_match,expected", [
    ([[0], [1]], [[], [1]], [0, 1], None, "did not list"),
    ([[0], []], [[0], []], [-1, -1], None, "blocking pair: suitor 0 and reviewer 0"),
    ([[0, 1], []], [[0], [0]], [1, -1], None, "blocking pair: suitor 0 and reviewer 0"),
    ([[0], [0]], [[0, 1]], [0, 0], None, "more than one suitor"),
    ([[0], []], [[0], []], [0, -1], [-1, -1], "inconsistent"),
    ([[0]], [[0]], [0, 0], None, "length"),
    ([[0]], [[0]], [3], None, "out-of-range"),
])
def test_checker_detects_errors(sp, rp, matching, reviewer_match, expected):
    assert expected in " | ".join(check_matching_partial(sp, rp, matching, reviewer_match))


def test_checker_accepts_unmatched_people():
    assert check_matching_partial([[0], []], [[0], []], [0, -1], [0, -1]) == []


def test_invalid_lists_rejected():
    assert validate_partial([[0, 0]], [[0]])
    assert validate_partial([[5]], [[0]])
    with pytest.raises(ValueError):
        gale_shapley_partial([[5]], [[0]])


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 20])
@pytest.mark.parametrize("seed", range(8))
def test_random_profiles_valid_and_one_to_one(n, seed, density):
    s, r = random_partial_profile(n, seed, density)
    res = gale_shapley_partial(s, r)
    assert_valid_stable_partial(s, r, res.suitor_match, res.reviewer_match)
    matched = [x for x in res.suitor_match if x != -1]
    assert len(matched) == len(set(matched))


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(8))
def test_suitor_optimal_and_same_people_matched(n, seed, density):
    s, r = random_partial_profile(n, seed, density)
    res = gale_shapley_partial(s, r)
    assert is_suitor_optimal_partial(s, r, res.suitor_match)
    matched_s = {i for i, x in enumerate(res.suitor_match) if x != -1}
    for other in enumerate_stable_matchings_partial(s, r):
        assert {i for i, x in enumerate(other) if x != -1} == matched_s


@pytest.mark.parametrize("n", [1, 2, 4, 7])
@pytest.mark.parametrize("seed", range(10))
def test_full_density_agrees_with_complete_list_reference(n, seed):
    s, r = random_partial_profile(n, seed, 1.0)
    a, b = gale_shapley_partial(s, r), gale_shapley(s, r)
    assert (a.suitor_match, a.reviewer_match) == (b.suitor_match, b.reviewer_match)


def test_score_encoding():
    assert to_scores_partial([[2, 0], []], 3) == [[0, -1, 1], [-1, -1, -1]]
    assert to_input_rows_partial([[2, 0], []], 3) == [[1, 0, 2], [0, 0, 0]]
    assert profile_to_scores([[1]], [[0], []]) == ([[-1, 0]], [[0], [-1]])


# ---- oblivious model on partial lists ----
@pytest.mark.parametrize("case", PARTIAL_HANDWRITTEN, ids=lambda c: c.name)
def test_model_handwritten_cases(case):
    ps, pr = profile_to_scores(case.suitor_prefs, case.reviewer_prefs)
    for res in (naive_oblivious_gs(ps, pr), optimized_partial(ps, pr)):
        assert res.suitor_match == case.expected_suitor_match
        assert_valid_stable_partial(case.suitor_prefs, case.reviewer_prefs, res.suitor_match, res.reviewer_match)


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("seed", range(15))
def test_model_equals_reference(n, seed, density):
    s, r = random_partial_profile(n, seed, density)
    ps, pr = profile_to_scores(s, r)
    ref = gale_shapley_partial(s, r)
    opt = optimized_partial(ps, pr)
    assert (opt.suitor_match, opt.reviewer_match) == (ref.suitor_match, ref.reviewer_match)
    assert naive_oblivious_gs(ps, pr, sweeps=n * n + 1).suitor_match == ref.suitor_match


def test_papers_stopping_rule_does_not_terminate_on_partial_lists():
    ps, pr = profile_to_scores([[], []], [[], []])
    with pytest.raises(RuntimeError):
        optimized_oblivious_gs(ps, pr)
    assert optimized_partial(ps, pr).suitor_match == [-1, -1]


# ---- unbalanced markets ----
def test_padding_adds_empty_dummies_only():
    sp, rp, n_s, n_r = pad_to_square([[0], [1, 0], [1], [0]], [[1, 3], [2]])
    assert (n_s, n_r) == (4, 2)
    assert rp[2:] == [[], []]
    assert all(x < 2 for lst in sp for x in lst)
    assert not validate_partial(sp, rp)


def test_unpad_rejects_matching_to_a_dummy():
    with pytest.raises(ValueError):
        unpad_matching([2, -1], [-1, -1], 2, 2)


@pytest.mark.parametrize("n_s,n_r", [(4, 2), (2, 4), (3, 3), (5, 1), (1, 5)])
@pytest.mark.parametrize("density", [0.4, 0.8, 1.0])
@pytest.mark.parametrize("seed", range(10))
def test_padded_model_equals_direct_reference(n_s, n_r, density, seed):
    s, r = random_partial_profile(n_s, seed, density, n_reviewers=n_r)
    ref = gale_shapley_partial(s, r)
    s_m, r_m = solve_rectangular_model(s, r)
    assert (s_m, r_m) == (ref.suitor_match, ref.reviewer_match)
    assert_valid_stable_partial(s, r, s_m, r_m)
