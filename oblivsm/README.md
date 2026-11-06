# CS670 Course Project: Privacy-Preserving Gale–Shapley Matching

Implementated Algorithms 3 and 4 from the paper *Fast and Secure Oblivious Stable Matching over Arithmetic Circuits*, and implemented a small extension to support partial preference lists and unbalanced markets.

What we built:
- **Plaintext Gale–Shapley** with a stability checker (`src/oblivsm/plaintext_gs.py`), used as the reference.
- **Algorithm 3** (naive) and **Algorithm 4** (optimized) as MP-SPDZ programs (`mpc/`).
- **Extension:** partial preference lists and unbalanced markets (`alg3_partial`, `alg4_partial`). Anyone can leave partners off their list and stay unmatched. Algorithm 4 needed a new stopping rule: stop when a sweep makes no progress.
- A cleartext model of Algorithms 3/4 (`src/oblivsm/model.py`), used to test the MPC programs.

## Run the tests

```bash
pip install -r requirements.txt
python3 -m pytest -m "not secure"        # no MP-SPDZ needed
```

For the tests on the real MPC programs, build MP-SPDZ first:

```bash
git clone https://github.com/data61/MP-SPDZ && cd MP-SPDZ
make -j8 shamir-party.x
Scripts/setup-ssl.sh 3
export MP_SPDZ_ROOT=$PWD
cd <this project>
python3 -m pytest -m secure
```

## Run the benchmark

```bash
python3 benchmarks/bench.py                                    # Algorithm 3 vs 4, n = 2 4 8 16
python3 benchmarks/bench.py --sizes 2 4 8 16 32 --seeds 0 1 2  # larger / more inputs
python3 benchmarks/bench.py --model-only                       # sweep counts only, no MP-SPDZ
```

Results go to `benchmarks/results/bench.csv`.
