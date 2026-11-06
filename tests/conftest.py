"""Shared pytest configuration: skip `secure` tests unless MP_SPDZ_ROOT is set."""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MP_SPDZ_ROOT"):
        return
    skip = pytest.mark.skip(reason="set MP_SPDZ_ROOT to a built MP-SPDZ checkout to run secure tests")
    for item in items:
        if "secure" in item.keywords:
            item.add_marker(skip)
