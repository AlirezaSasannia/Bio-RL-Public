"""Pre-registration: freeze a spec before running, refuse to run if it changed afterwards.

register(spec) writes preregistrations/<id>.yaml once, with a SHA-256 of its
content and a UTC timestamp in preregistrations/registry.csv. A later call with
a different spec for the same id raises, so predictions cannot drift after the
results are seen.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
PREREG = ROOT / "preregistrations"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def register(spec: dict) -> str:
    PREREG.mkdir(exist_ok=True)
    text = yaml.safe_dump(spec, sort_keys=True, allow_unicode=True)
    h = _hash(text)
    path = PREREG / f"{spec['id']}.yaml"
    reg = PREREG / "registry.csv"
    if path.exists():
        if _hash(path.read_text()) != h:
            raise RuntimeError(f"Pre-registration {spec['id']} already exists with different content; "
                               "create a new id instead of editing a registered spec.")
        return h
    path.write_text(text)
    new = not reg.exists()
    with reg.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "sha256", "registered_utc"])
        w.writerow([spec["id"], h, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")])
    return h
