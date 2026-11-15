# Bio-RL

Code and pre-registrations for a study of **biologically inspired control of a reinforcement learner in
non-stationary worlds**. A tabular agent lives in a small gridworld whose regime (food, danger, volatility, control)
changes without warning. On top of the agent's ordinary learning rule, a global layer reads the agent's own internal
signals and re-tunes its learning rate, exploration and risk attitude during its life. The repository contains the
simulator, the agents and global layers, the pre-registered experiment pipeline, and the registry of
pre-registrations.

> **Note:** Certain manuscript files and full reproduction logs are temporarily withheld while the manuscript is
> under review and will be fully released upon publication.

## Related repository

- **Bio-RL platform** (interactive web viewer of the simulator, separate repository): <https://github.com/amirjahantab/Bio_RL_platform_public>

## Layout

| Path | Contents |
|---|---|
| `04_code/src/bio_rl/` | Simulator (gridworld, needs, appraisal signals, learners, global modulation layers) |
| `04_code/experiments/single/` | Pre-registered single-agent experiment stages, statistical analyses, validation scripts |
| `04_code/preregistrations/` | Registered specifications (`*.yaml`) and `registry.csv` (id, SHA-256, UTC registration time) |
| `04_code/configs/` | Architecture configurations |
| `04_code/tests/` | Unit tests |

## Install and test

    cd 04_code
    python -m venv .venv
    .venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    python -m pytest -q tests

## Pre-registration

Every experiment stage was registered before its first run with `bio_rl.battery.prereg.register(spec)`, which stores
the spec as YAML and appends its SHA-256 and UTC time to `04_code/preregistrations/registry.csv`. The files in
`04_code/preregistrations/` are byte-identical copies of the registered specifications. The complete original
version history, including the time-ordered registration commits, is preserved and will be archived (Zenodo) upon
publication.

## Authors

- **Amirreza Jahantab** 
- **Alireza Sasannia**
- **Aytan Dalfardi**

See the commit history for individual contributions.

## License

Code: MIT. Pre-registrations and documentation: CC BY 4.0. See `LICENSE`.
