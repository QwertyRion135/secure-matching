"""Runs the .mpc programs on an MP-SPDZ checkout: assemble, compile, write inputs, run, parse.

Environment variables:
    MP_SPDZ_ROOT             path of a built MP-SPDZ checkout (required)
    OBLIVSM_PROTOCOL_SCRIPT  script in <root>/Scripts to run all parties (default shamir.sh)
    OBLIVSM_COMPILE_FLAGS    extra flags for compile.py (default none)
    OBLIVSM_TIMEOUT          seconds per compile / run (default 3600)

Party 0 provides the suitors' preferences, party 1 the reviewers'.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .partial import to_input_rows_partial
from .profiles import to_scores

MPC_DIR = Path(__file__).resolve().parents[2] / "mpc"

# program key -> source files, concatenated in this order
PROGRAMS: dict[str, list[str]] = {
    "primitives_test": ["primitives.mpc", "test_primitives.mpc"],
    "alg3": ["primitives.mpc", "gs_common.mpc", "alg3_naive.mpc"],
    "alg4": ["primitives.mpc", "gs_common.mpc", "alg4_optimized.mpc"],
    "alg3_partial": ["primitives.mpc", "gs_common.mpc", "alg3_partial.mpc"],
    "alg4_partial": ["primitives.mpc", "gs_common.mpc", "alg4_partial.mpc"],
}


class MPSpdzError(RuntimeError):
    pass


@dataclass
class RunResult:
    suitor_match: list[int]
    reviewer_match: list[int]
    sweeps: Optional[int]            # printed by Algorithm 4 only
    compile_seconds: float
    wall_seconds: float
    mpc_seconds: Optional[float]     # "Time = ..." from MP-SPDZ
    data_sent_mb: Optional[float]    # party 0
    global_data_mb: Optional[float]  # all parties, if printed
    rounds: Optional[int]
    output: str


def mp_spdz_root() -> Optional[Path]:
    root = os.environ.get("MP_SPDZ_ROOT")
    return Path(root).expanduser().resolve() if root else None


def _need_root(root: Optional[Path]) -> Path:
    root = root or mp_spdz_root()
    if root is None:
        raise MPSpdzError("set MP_SPDZ_ROOT to a built MP-SPDZ checkout")
    return root


def install_program(root: Path, key: str) -> str:
    """Write the assembled program into Programs/Source and return its name."""
    name = f"oblivsm_{key}"
    target = root / "Programs" / "Source" / f"{name}.mpc"
    target.parent.mkdir(parents=True, exist_ok=True)
    source = "\n\n".join((MPC_DIR / part).read_text() for part in PROGRAMS[key])
    if not target.exists() or target.read_text() != source:
        target.write_text(source)
    return name


def _run(cmd: Sequence[str], cwd: Path, what: str) -> str:
    timeout = float(os.environ.get("OBLIVSM_TIMEOUT", "3600"))
    try:
        proc = subprocess.run(list(cmd), cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise MPSpdzError(f"{what}: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        raise MPSpdzError(f"{what} failed (exit {proc.returncode}); last output:\n{tail}")
    return proc.stdout


def compile_program(root: Path, name: str, args: Sequence[object] = ()) -> float:
    flags = shlex.split(os.environ.get("OBLIVSM_COMPILE_FLAGS", ""))
    start = time.perf_counter()
    _run([sys.executable, "./compile.py", *flags, name, *map(str, args)], root, "compile")
    return time.perf_counter() - start


def write_inputs(root: Path, suitor_prefs, reviewer_prefs, partial: bool = False) -> None:
    """Player-Data/Input-P0-0 = suitors' scores, Input-P1-0 = reviewers' scores."""
    n = len(suitor_prefs)
    data_dir = root / "Player-Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for party, prefs in ((0, suitor_prefs), (1, reviewer_prefs)):
        rows = to_input_rows_partial(prefs, n) if partial else to_scores(prefs)
        (data_dir / f"Input-P{party}-0").write_text("\n".join(" ".join(map(str, r)) for r in rows) + "\n")


def _run_parties(root: Path, schedule: str) -> tuple[str, float]:
    script = os.environ.get("OBLIVSM_PROTOCOL_SCRIPT", "shamir.sh")
    start = time.perf_counter()
    out = _run([str(root / "Scripts" / script), schedule], root, "run")
    return out, time.perf_counter() - start


def parse_stats(text: str):
    """(time_s, party0_data_MB, global_data_MB, rounds); None where MP-SPDZ doesn't print it."""
    t = re.search(r"Time = ([\d.]+) seconds", text)
    sent = re.search(r"Data sent = ([\d.]+) MB in (\d+) rounds", text)
    g = re.search(r"Global data sent = ([\d.]+) MB", text)
    return (float(t.group(1)) if t else None,
            float(sent.group(1)) if sent else None,
            float(g.group(1)) if g else None,
            int(sent.group(2)) if sent else None)


def _normalise(value: int) -> int:
    # unmatched is revealed as -1, or as a huge field element on some builds
    return -1 if value < 0 or value >= 2**32 else value


def parse_matching(text: str, n: int) -> tuple[list[int], list[int]]:
    suitors, reviewers = [None] * n, [None] * n
    for kind, idx, val in re.findall(r"OBLIVSM_(SUITOR|REVIEWER)\s+(\d+)\s+(-?\d+)", text):
        (suitors if kind == "SUITOR" else reviewers)[int(idx)] = _normalise(int(val))
    if None in suitors or None in reviewers:
        raise MPSpdzError("matching missing from MP-SPDZ output:\n" + text[-1500:])
    return suitors, reviewers


def parse_prims(text: str) -> dict[str, int]:
    """Values printed by test_primitives.mpc (-1 may appear as a huge field element)."""
    return {name: -1 if int(v) < 0 or int(v) >= 2**60 else int(v)
            for name, v in re.findall(r"OBLIVSM_PRIM\s+(\w+)\s+(-?\d+)", text)}


def run_primitive_checks(root: Optional[Path] = None) -> dict[str, int]:
    root = _need_root(root)
    name = install_program(root, "primitives_test")
    compile_program(root, name)
    out, _ = _run_parties(root, name)
    return parse_prims(out)


def run_matching(key: str, suitor_prefs, reviewer_prefs, sweeps: Optional[int] = None,
                 root: Optional[Path] = None, compile_cache: Optional[set] = None) -> RunResult:
    """Run alg3, alg4, alg3_partial or alg4_partial on a profile (ordered lists, square market).

    `sweeps` overrides Algorithm 3's sweep count. `compile_cache` (a set owned by the
    caller) avoids recompiling the same program and size.
    """
    if key not in ("alg3", "alg4", "alg3_partial", "alg4_partial"):
        raise ValueError(f"unknown program {key!r}")
    root = _need_root(root)
    n = len(suitor_prefs)
    args: list[object] = [n]
    if key.startswith("alg3") and sweeps is not None:
        args.append(sweeps)

    name = install_program(root, key)
    schedule = "-".join([name, *map(str, args)])      # compile.py's schedule name
    compile_s = 0.0
    if compile_cache is None or schedule not in compile_cache:
        compile_s = compile_program(root, name, args)
        if compile_cache is not None:
            compile_cache.add(schedule)

    write_inputs(root, suitor_prefs, reviewer_prefs, partial=key.endswith("_partial"))
    out, wall = _run_parties(root, schedule)

    s_m, r_m = parse_matching(out, n)
    m = re.search(r"OBLIVSM_SWEEPS\s+(\d+)", out)
    time_s, sent_mb, global_mb, rounds = parse_stats(out)
    return RunResult(s_m, r_m, int(m.group(1)) if m else None, compile_s, wall,
                     time_s, sent_mb, global_mb, rounds, out)
