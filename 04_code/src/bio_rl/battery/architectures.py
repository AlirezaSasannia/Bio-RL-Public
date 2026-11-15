"""Load architecture versions from configs/architectures/*.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

from bio_rl.endocrine.hormones import EndocrineConfig
from bio_rl.learning.agents import AgentConfig
from bio_rl.needs.homeostasis import NeedConfig

ARCH_DIR = Path(__file__).resolve().parents[3] / "configs" / "architectures"


def load(name: str) -> dict:
    spec = yaml.safe_load((ARCH_DIR / f"{name}.yaml").read_text())
    if "base" in spec:
        base = load(spec["base"])
        base["endocrine"] = {**base.get("endocrine", {}), **spec.get("endocrine", {})}
        return base
    return spec


def agent_config(name: str, social: bool = False, **endo_overrides) -> AgentConfig:
    spec = load(name)
    needs = NeedConfig(social=social)
    endo = EndocrineConfig(**{**spec.get("endocrine", {}), **endo_overrides})
    return AgentConfig(kind=spec["kind"], needs=needs, endocrine=endo)
