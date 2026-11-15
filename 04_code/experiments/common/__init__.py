"""Shared helpers for experiments 4-6."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
LOGS = ROOT / "results" / "logs"

from bio_rl.endocrine.hormones import EndocrineConfig  # noqa: E402

# endocrine channel presets: start from "all off" and switch channels on
OFF = dict(ne_to_alpha=0.0, ne_to_temp=0.0, c_to_safety=0.0, c_alpha_up=0.0, c_alpha_down=0.0,
           c_to_temp=0.0, c_alpha_lower=0.0, c_setpoint=0.0)


def endo(**on) -> EndocrineConfig:
    kw = dict(OFF)
    kw.update(on)
    return EndocrineConfig(**kw)
