# Running stage C3 on a Linux GPU server

Stage C3 is the view learner of stage C2 (`world.py`, learner `dqnv`) with 8 new independent bases, a GRU
baseline and an online meta-gradient baseline. Script: `experiments/single/stage_c3.py`. The design is in the registrations in
`preregistrations/`.

## State at hand-over (2026-10-21)

| Step | Status |
|---|---|
| C3-0 bases (`single_stageC3_base`, sha 1317a0fe) | registered; paused after 1 of 30 generations on the RTX 2050 |
| C3 arms (`single_stageC3_independent_bases_gru_metagrad`) | **not registered yet**; `stage_c3.py arms` registers it when it starts |
| test, savings, report | not run |

**Restart C3-0 on the server rather than resume.** Only one generation (~7 min) was done. A resumed run would mix
two GPU models within one optimisation, and the Windows checkpoint cannot be resumed anyway:
`_resume.cma_lockstep` checks the run paths stored in the checkpoint, and those are absolute Windows paths.
`*.ckpt` files are git-ignored, so a fresh clone has no checkpoint and the base step starts from generation 0. The
registration covers the design and seeds, not the hardware, so it does not change.

## Requirements

- **OS:** Linux x86-64 with an NVIDIA GPU, at least 4 GB of memory (8 GB or more recommended), and a recent NVIDIA
  driver (`nvidia-smi` must work).
- **Python:** 3.12 or 3.13.
- **Disk:** about 12 GB free, made up of:
  - venv with the CUDA build of torch: ~6 GB
  - evaluation caches: ~3 GB (test ~0.8 GB, savings ~2.2 GB)
  - logs, JSON and CSV: < 100 MB
- **RAM:** 8 GB for one process; 16 GB if you run three processes in parallel. Each process uses ~2–3 GB of host
  RAM, mostly the CUDA context and traces.
- **GPU memory per process:** ~0.7 GB for the dqnv arms and ~1.7 GB for the GRU arms. Each simulator call runs 2,304
  agents (4 bases × 12 candidates × 48) in training and 2,048 in test/savings.

## 1. Environment

```bash
git clone <repo-url> Bio-RL          # or: cd Bio-RL && git pull
cd Bio-RL
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy==2.5.4 scipy==1.18.1 pandas==3.0.6 cma==4.5.0 PyYAML==6.0.3 matplotlib==3.11.2
```

Install the torch build that matches the driver (check the CUDA version shown by `nvidia-smi`):

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # driver for CUDA 12.8+ (R570+)
# older driver: use .../whl/cu126 (CUDA 12.6, R560+) or .../whl/cu118
```

Check it:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

These are the versions used on the Windows machine, where torch was 2.11.0+cu128 on driver 560.94.

## 2. Checks before the real run (about 25 minutes)

All commands run from `04_code/experiments/single` with the venv active.

**(a) Code check on development seeds.** This uses 1 generation, 600-step lives and tiny batches, writes to `/tmp`
and registers nothing. It must end with `=== C3 finished`:

```bash
cd 04_code/experiments/single
C3_SMOKE=/tmp/c3smoke python stage_c3.py all 2>&1 | tail -5
rm -rf /tmp/c3smoke
```

**(b) Timing on development seeds.** This prints ms per 6000-step agent-life:

```bash
python bench_c3.py 300 2304 dqnv,metagrad,gru
```

Put the three numbers (`m_d` for dqnv, `m_m` for metagrad, `m_g` for gru, all at 2,304 agents) into:

    total seconds  ~  138 * m_d  +  108 * (5 * m_d + m_m + 2 * m_g)

The first term is the bases. The second covers the arms, which are 92 × (5 m_d + m_m + 2 m_g); test ≈ 4.5 ×
(…); and savings ≈ 11.5 × (…). Measured on the RTX 2050: m_d = 87, m_m = 122, m_g = 276, which gives ~36 h.

## 3. Run

Use `tmux` (or `nohup … &`) so the run survives a dropped connection. Every step is resumable: re-running the same
command continues from the last completed generation (checkpoint per generation) or the last cached evaluation.

**Bases** (30 generations; on the RTX 2050, 397 s per generation):

```bash
cd 04_code/experiments/single
python -u stage_c3.py base > ../../results/logs/single/stageC3_base.log 2> ../../results/logs/single/stageC3_base.err
```

**Check the bases before registering the arms.** The last 8 lines of the log list each base's parameters and which
of them are within 2% of a bound:

```bash
grep "^.*base [0-7] " ../../results/logs/single/stageC3_base.log
```

If a base sits on a bound other than the simulator cap eps0 = 0.5, stop and write an amendment before running the
arms (see CLAUDE.md, scientific rule 1).

**Arms.** `arms` registers the arm spec on its first call; a second process sees the same spec and registers
nothing new. You can use one process (sequential) or, on a GPU with ≥ 8 GB, three processes in parallel. Results
do not depend on the split, because every arm has its own seeds, checkpoint and output files.

```bash
L=../../results/logs/single
# one process:
python -u stage_c3.py arms > $L/stageC3_arms.log 2> $L/stageC3_arms.err
# or three processes (start the first, wait until "registered" appears in its log, then start the others):
python -u stage_c3.py arms gru      > $L/stageC3_arms_gru.log      2> $L/stageC3_arms_gru.err &
python -u stage_c3.py arms gru_aug  > $L/stageC3_arms_gru_aug.log  2> $L/stageC3_arms_gru_aug.err &
python -u stage_c3.py arms control,aug,film,generic,bias,metagrad > $L/stageC3_arms_ff.log 2> $L/stageC3_arms_ff.err &
```

**Test, savings protocol and report**, after all 64 arm runs exist (`ls $L/stageC3_opt/*.json | wc -l` gives 64):

```bash
python -u stage_c3.py test    > $L/stageC3_test.log    2> $L/stageC3_test.err
python -u stage_c3.py savings > $L/stageC3_savings.log 2> $L/stageC3_savings.err
python -u stage_c3.py report  > $L/stageC3_report.log  2> $L/stageC3_report.err
```

**Monitoring:**
- `tail -f $L/stageC3_*.log` shows one line per generation with the time taken.
- `nvidia-smi -l 60` shows GPU load.

To stop a run, kill it by PID (`ps aux | grep stage_c3`, then `kill <PID>`). Never use `pkill -f` with a pattern
that also matches your own shell.

## 4. Expected time per GPU class

| GPU | Basis | Full C3 (bases + arms + test + savings) |
|---|---|---|
| RTX 2050 laptop, 4 GB | measured: 397 s per base generation | ~36 h, sequential only |
| RTX 3070 / 3080 / A4000 class | ~3× the 2050, from stage C2 timings on a 3070 | ~12–14 h; ~8 h with 3 processes |
| RTX 4090 / A100 / H100 | extrapolated | ~8–12 h sequential; ~4–6 h with 3 processes |

Rows two and three are estimates, not measurements; run `bench_c3.py` and the formula above before planning.
Sequential runs on fast GPUs are limited by per-step kernel-launch overhead (one call of 2,304 agents takes 6,000
small steps, and a fast CPU and Linux help). That is why parallel processes on a large GPU give more than a faster
card alone.

## 5. Afterwards

**Results to keep, all in git** (under `results/logs/single/`):
- `stageC3_base/*.json` and `stageC3_opt/*.json`
- `stageC3_test_raw.csv`, `stageC3_tests.csv`, `stageC3_summary.csv`, `stageC3_base_means.csv`
- `stageC3_params.csv`
- `stageC3_savings_raw.csv`, `stageC3_savings_tests.csv`, `stageC3_savings_summary.csv`
- the `.err` files

Logs (`*.log`), caches (`stageC3_cache/`) and checkpoints (`*.ckpt`) are git-ignored. Commit on the server, or copy
the files back with `rsync`.

Record in `results/decision_log.md`:
- the server's GPU model, driver, torch/CUDA build and Python version
- the measured time per generation
- the date

Reproducibility: results are bit-reproducible on the same GPU model, driver and torch build. Across GPU models,
floating-point differences make individual numbers differ slightly, but not the statistics.

---

# Stage C4 (second environment E2b) on a Linux GPU server — added 2026-10-30

Only if `results/logs/single/stageE2b_0/gate_summary.csv` says `gate = PASS` (the script refuses otherwise). The full
decision tree, reading of the outputs and the paper consequences are in `04_code/RUNBOOK_fa.md` (Persian).

1. Environment as in section 1 above. Commit and pull `stageE2b_0/*.csv` and `*.json` (the caches are not needed).
2. Code check (about 10 min on an RTX 2050; writes to /tmp, registers nothing; uses the E2b smoke output):
   ```bash
   cd 04_code/experiments/single
   E2B_SMOKE=/tmp/e2b python stage_e2b_0.py all > /tmp/e2b.log 2>&1
   C4_SMOKE=/tmp/c4 C4_E2B=/tmp/e2b python stage_c4.py all 2>&1 | tail -3      # must end with "=== C4 finished"
   rm -rf /tmp/e2b /tmp/c4
   ```
3. Timing and memory: `python bench_c4.py 300 2304,4608 dqnv,gru,ff_big`. RTX 2050 reference at 2304 agents:
   dqnv 194, ff_big 435, gru 621 ms per 9000-step agent-life; peak GPU 0.9 / 2.2 / 2.3 GB.
4. Choose `C4_BPC` (bases per GPU call; 4 = 2304 agents per training call, 8 = 4608) from the peak memory, BEFORE
   registering; it is part of both specs and must not change during the stage.
5. Register C4-0 and commit, then run the chain detached:
   ```bash
   export C4_BPC=4
   python stage_c4.py register_base && git add ../../preregistrations && git commit -m "Register C4-0"
   nohup bash run_c4_chain.sh > /dev/null 2>&1 &
   ```
   The chain registers the arms spec after the bases are frozen (commit that yaml when it appears). To run arms in
   parallel processes, replace the single `arms` line (see the comment in `run_c4_chain.sh`); start them only after
   `register_arms` has run. Re-run any step to resume (per-generation checkpoints, evaluation cache). Checkpoints hold
   absolute paths: do not move the checkout during a run, and do not resume a Windows checkpoint on Linux.
6. Expected time: about 89 h on the RTX 2050, about 30 h on an RTX 3070, about 15 h on a 4090/A100 (one process;
   parallel processes faster), +-30%.
7. Outputs: `results/logs/single/stageC4_{tests,summary,base_means,params,test_raw}.csv`, `stageC4_base/`,
   `stageC4_opt/`. Commit those (not `stageC4_cache/`, not `*.ckpt`).
