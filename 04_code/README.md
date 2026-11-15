# Bio-RL code

- `src/bio_rl/fast/`: vectorised simulator (`adaptive.py`: learners and global layers; `regimes.py`: hidden regime
  schedules; `popsim.py`; `adaptive_search.py`: search spaces).
- `src/bio_rl/torchsim/`: GPU simulator for the network learners.
- `src/bio_rl/battery/prereg.py`: pre-registration helper (`register(spec)`).
- `experiments/single/stage_*.py`: one script per registered stage. Each stage optimises its arms with an equal
  CMA-ES budget, uses common random numbers across arms and evaluates once on a test set that was never used before.
- `experiments/single/` analyses (`tost_b3.py`, `per_regime_b7.py`, `mechanism.py`, `clamp.py`, `reanalysis_*.py`)
  and validation scripts (`validate_*.py`).

Stage scripts write to `results/logs/single/`. The logs of the registered runs and the frozen optimised parameters
are withheld until publication, so stages that start from frozen bases need the base stage to be run first.

Workers: `max(1, os.cpu_count() - 1)`; results do not depend on the pool size (seeds are per job).
