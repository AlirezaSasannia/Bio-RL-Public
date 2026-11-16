import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bio_rl.endocrine.hormones import Endocrine, EndocrineConfig
from bio_rl.needs.homeostasis import Needs


def test_hunger_keeps_signalling_at_zero_energy():
    n = Needs()
    n.n = np.array([0.0, 1.0])
    dn = n.update({"ate": False, "damage": False, "in_shelter": False, "moved": True})
    assert dn[0] < 0 and n.n[0] == 0.0


def test_cortisol_rises_only_when_threat_is_uncontrollable():
    hi, lo = Endocrine(EndocrineConfig(delay=1)), Endocrine(EndocrineConfig(delay=1))
    for _ in range(50):
        hi.step(0.0, threat=1.0, K=0.0, U=0.0)
        lo.step(0.0, threat=1.0, K=1.0, U=0.0)
    assert hi.C > 0.1 and lo.C == 0.0


def test_cortisol_returns_to_baseline():
    e = Endocrine(EndocrineConfig(delay=1))
    for _ in range(20):
        e.step(0.0, 1.0, 0.0, 0.0)
    peak = e.C
    for _ in range(1000):
        e.step(0.0, 0.0, 1.0, 0.0)
    assert e.C < 0.05 * peak


def test_contingency_controllability():
    from bio_rl.appraisal.appraisal import Appraisal
    esc, yok = Appraisal(4, kind="contingency"), Appraisal(4, kind="contingency")
    assert esc.Kp == 0.5
    rng = np.random.default_rng(0)
    for t in range(3000):
        esc.update(0, rng.random() < 0.35)          # damage only in one avoidable place
        esc.update(1, False)
        yok.update(int(rng.integers(4)), rng.random() < 0.05)   # damage anywhere
    assert esc.Kp > 0.8 and yok.Kp < 0.45   # random hits briefly mark cells as dangerous, so yoked Kp stays above 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
