"""Plaintext Gale-Shapley reference and checker."""
import pytest

from oblivsm.plaintext_gs import (
    assert_valid_stable, check_matching, gale_shapley, is_suitor_optimal, validate_preferences,
)
from oblivsm.profiles import HANDWRITTEN, immediate_match_case, long_sequence_case, random_profile


@pytest.mark.parametrize("case", HANDWRITTEN, ids=lambda c: c.name)
def test_handwritten_expected_matching(case):
    res = gale_shapley(case.suitor_prefs, case.reviewer_prefs)
    assert res.suitor_match == case.expected_suitor_match
    assert_valid_stable(case.suitor_prefs, case.reviewer_prefs, res.suitor_match, res.reviewer_match)


def test_two_stable_matchings_returns_suitor_optimal():
    s, r = [[0, 1], [1, 0]], [[1, 0], [0, 1]]
    assert not check_matching(s, r, [0, 1])
    assert not check_matching(s, r, [1, 0])
    assert gale_shapley(s, r).suitor_match == [0, 1]


def test_immediate_match_needs_n_proposals():
    case = immediate_match_case(5)
    res = gale_shapley(case.suitor_prefs, case.reviewer_prefs)
    assert res.suitor_match == list(range(5))
    assert res.proposals == 5


@pytest.mark.parametrize("n", [2, 3, 5, 8, 16])
def test_long_proposal_sequence(n):
    case = long_sequence_case(n)
    res = gale_shapley(case.suitor_prefs, case.reviewer_prefs)
    assert res.suitor_match == case.expected_suitor_match
    assert res.proposals == n * (n + 1) // 2


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 25])
@pytest.mark.parametrize("seed", range(10))
def test_random_profiles_stable_and_one_to_one(n, seed):
    s, r = random_profile(n, seed)
    res = gale_shapley(s, r)
    assert_valid_stable(s, r, res.suitor_match, res.reviewer_match)
    assert sorted(res.suitor_match) == list(range(n))
    assert res.proposals <= n * n - n + 1


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", range(15))
def test_result_is_suitor_optimal(n, seed):
    s, r = random_profile(n, seed)
    assert is_suitor_optimal(s, r, gale_shapley(s, r).suitor_match)


def test_invalid_profiles_rejected():
    assert validate_preferences([[0, 0], [0, 1]], [[0, 1], [0, 1]])
    assert validate_preferences([[0, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        gale_shapley([[0, 2], [0, 1]], [[0, 1], [0, 1]])


# ---- the checker must catch bad matchings ----
SAME = ([[0, 1], [0, 1]], [[0, 1], [0, 1]])


def test_checker_accepts_valid_matching():
    assert check_matching(*SAME, [0, 1], [0, 1]) == []


@pytest.mark.parametrize("matching,reviewer_match,expected", [
    ([1, 0], None, "blocking pair: suitor 0 and reviewer 0"),
    ([0, 0], None, "more than one suitor"),
    ([-1, 0], None, "unmatched"),
    ([0, 5], None, "out-of-range"),
    ([0, 1], [1, 0], "inconsistent"),
    ([0], None, "length"),
])
def test_checker_detects_errors(matching, reviewer_match, expected):
    assert expected in " | ".join(check_matching(*SAME, matching, reviewer_match))


def test_checker_rejects_invalid_lists():
    assert "permutation" in " | ".join(check_matching([[0, 0], [0, 1]], [[0, 1], [0, 1]], [0, 1]))
