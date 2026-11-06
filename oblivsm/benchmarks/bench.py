#!/usr/bin/env python3
"""Benchmark Algorithm 3 vs Algorithm 4 over growing market sizes.

    export MP_SPDZ_ROOT=/path/to/MP-SPDZ
    python benchmarks/bench.py                                  # n = 2 4 8 16
    python benchmarks/bench.py --sizes 2 4 8 16 32 --seeds 0 1 2 --repeats 3
    python benchmarks/bench.py --model-only --sizes 2 4 8 16 32 # sweep counts only, no MP-SPDZ

Inputs are generated from fixed seeds (profiles.benchmark_profile), so every machine uses
the same ones. Results are written to benchmarks/results/bench.csv.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oblivsm.model import naive_oblivious_gs, optimized_oblivious_gs  # noqa: E402
from oblivsm.mpspdz import MPSpdzError, mp_spdz_root, run_matching  # noqa: E402
from oblivsm.plaintext_gs import check_matching, gale_shapley  # noqa: E402
from oblivsm.profiles import benchmark_profile, to_scores  # noqa: E402

FIELDS = ["n", "seed", "alg", "repeat", "compile_s", "wall_s", "mpc_s", "data_mb",
          "global_mb", "rounds", "sweeps", "matches_ref", "valid_stable"]


def model_only(sizes, seeds):
    print(f"{'n':>5} {'seed':>5} {'sweeps (Alg.4)':>15} {'Alg.3 complete':>15}")
    for n in sizes:
        for seed in seeds:
            s, r = benchmark_profile(n, seed)
            ps, pr = to_scores(s), to_scores(r)
            print(f"{n:>5} {seed:>5} {optimized_oblivious_gs(ps, pr).sweeps:>15} "
                  f"{str(naive_oblivious_gs(ps, pr).complete):>15}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "bench.csv")
    ap.add_argument("--model-only", action="store_true")
    args = ap.parse_args()

    if args.model_only:
        model_only(args.sizes, args.seeds)
        return 0
    if mp_spdz_root() is None:
        print("error: set MP_SPDZ_ROOT (or use --model-only)", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, compiled = [], set()
    for n in args.sizes:
        for seed in args.seeds:
            s, r = benchmark_profile(n, seed)
            ref = gale_shapley(s, r).suitor_match
            for alg in ("alg3", "alg4"):
                for rep in range(args.repeats):
                    try:
                        run = run_matching(alg, s, r, compile_cache=compiled)
                    except MPSpdzError as exc:
                        print(f"n={n} seed={seed} {alg}: FAILED\n{exc}", file=sys.stderr)
                        continue
                    row = dict(n=n, seed=seed, alg=alg, repeat=rep,
                               compile_s=round(run.compile_seconds, 3), wall_s=round(run.wall_seconds, 3),
                               mpc_s=run.mpc_seconds, data_mb=run.data_sent_mb, global_mb=run.global_data_mb,
                               rounds=run.rounds, sweeps=run.sweeps,
                               matches_ref=run.suitor_match == ref,
                               valid_stable=not check_matching(s, r, run.suitor_match, run.reviewer_match))
                    rows.append(row)
                    print(row, flush=True)
                    with args.out.open("w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=FIELDS)
                        w.writeheader()
                        w.writerows(rows)

    print(f"\n{'n':>5} {'alg':>5} {'wall_s':>10} {'data_mb':>10} {'rounds':>10}   (means; csv: {args.out})")
    for n in args.sizes:
        for alg in ("alg3", "alg4"):
            sel = [x for x in rows if x["n"] == n and x["alg"] == alg]
            if sel:
                def mean(key):
                    vals = [x[key] for x in sel if x[key] is not None]
                    return f"{statistics.mean(vals):.3f}" if vals else "n/a"
                print(f"{n:>5} {alg:>5} {mean('wall_s'):>10} {mean('data_mb'):>10} {mean('rounds'):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
