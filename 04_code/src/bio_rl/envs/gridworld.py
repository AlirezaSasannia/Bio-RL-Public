"""Foraging gridworld with food, a shelter and threats.

The environment only reports *events* (food eaten, damage taken, being in the
shelter). It does not hand out reward: what an event is worth is decided by the
agent's internal needs (see ``bio_rl.needs``) or, for the fixed-reward
baseline, by a designer-specified reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# actions: stay, up, down, left, right
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
N_ACTIONS = len(MOVES)


@dataclass
class Layout:
    """Positions of food, shelter and threat cells on a ``size`` x ``size`` grid."""

    food: list[tuple[int, int]]
    shelter: tuple[int, int]
    threat: list[tuple[int, int]]
    size: int = 8
    companion: tuple[int, int] | None = None   # a friendly agent; being on its cell = social contact


# Training world and a held-out novel world: food, shelter and threat all move.
LAYOUT_A = Layout(
    food=[(1, 1), (1, 6)],
    shelter=(6, 1),
    threat=[(3, 3), (3, 4), (4, 3), (4, 4), (2, 4), (4, 5)],
)
LAYOUT_B = Layout(
    food=[(6, 6), (5, 2)],
    shelter=(1, 6),
    threat=[(5, 4), (5, 5), (4, 5), (6, 4), (3, 2), (4, 2)],
)


def make_layout(seed: int, size: int = 8) -> Layout:
    """Procedural world: 1 shelter, 2 food cells, a cluster of 6 threat cells."""
    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(size) for c in range(size)]
    idx = rng.permutation(len(cells))
    shelter = cells[idx[0]]
    food = []
    for i in idx[1:]:
        cand = cells[i]
        if abs(cand[0] - shelter[0]) + abs(cand[1] - shelter[1]) >= 4 and all(
                abs(cand[0] - f[0]) + abs(cand[1] - f[1]) >= 4 for f in food):
            food.append(cand)
        if len(food) == 2:
            break
    return with_threat(Layout(food=food, shelter=shelter, threat=[], size=size), seed + 7919)


def make_layout_e2(seed: int, size: int = 12, n_food: int = 3, n_clusters: int = 2, cluster: int = 7,
                   min_dist: int = 6) -> Layout:
    """Second environment E2: 1 shelter, 3 food cells (pairwise and from the shelter at least min_dist steps apart),
    2 non-overlapping threat clusters of 7 cells, each drawn inside a 3 x 3 block that avoids food and shelter."""
    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(100):
        idx = rng.permutation(len(cells))
        shelter = cells[idx[0]]
        food = []
        for i in idx[1:]:
            cand = cells[i]
            if all(abs(cand[0] - p[0]) + abs(cand[1] - p[1]) >= min_dist for p in food + [shelter]):
                food.append(cand)
            if len(food) == n_food:
                break
        if len(food) == n_food:
            break
    blocked = set(food) | {tuple(shelter)}
    threat = []
    for _ in range(n_clusters):
        for _ in range(500):
            cr, cc = rng.integers(1, size - 1, size=2)
            ring = [(int(cr + dr), int(cc + dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            ring = [x for x in ring if x not in blocked]
            if len(ring) >= cluster:
                pick = [ring[i] for i in rng.choice(len(ring), size=cluster, replace=False)]
                threat += pick
                blocked |= {(r + dr, c + dc) for r, c in pick for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                break
    return Layout(food=food, shelter=tuple(shelter), threat=threat, size=size)


def with_threat(layout: Layout, seed: int, n: int = 6) -> Layout:
    """Same food and shelter, a new threat cluster placed between them."""
    rng = np.random.default_rng(seed)
    size = layout.size
    blocked = set(layout.food) | {tuple(layout.shelter)}
    for _ in range(200):
        cr, cc = rng.integers(1, size - 1, size=2)
        ring = [(cr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        ring = [x for x in ring if 0 <= x[0] < size and 0 <= x[1] < size and x not in blocked]
        if len(ring) >= n:
            pick = rng.choice(len(ring), size=n, replace=False)
            return Layout(food=list(layout.food), shelter=tuple(layout.shelter),
                          threat=[tuple(int(v) for v in ring[i]) for i in pick], size=size)
    return layout


def with_companion(layout: Layout, seed: int) -> Layout:
    """Add a companion on a free cell at least 3 steps from the shelter."""
    rng = np.random.default_rng(seed)
    taken = set(layout.food) | set(layout.threat) | {tuple(layout.shelter)}
    free = [(r, c) for r in range(layout.size) for c in range(layout.size)
            if (r, c) not in taken and abs(r - layout.shelter[0]) + abs(c - layout.shelter[1]) >= 3]
    comp = free[int(rng.integers(len(free)))]
    return Layout(food=list(layout.food), shelter=tuple(layout.shelter), threat=list(layout.threat),
                  size=layout.size, companion=comp)


@dataclass
class EnvConfig:
    regrow_steps: int = 30          # food reappears this many steps after being eaten
    p_damage_zone: float = 0.35     # damage probability per step inside a threat cell
    uncontrollable: bool = False    # if True, damage is random everywhere (same mean rate)
    p_damage_uncontrollable: float = 0.04
    no_damage: bool = False                   # 'none' group of the triadic design
    damage_schedule: object = None            # yoked group: array of bools, damage at step t regardless of action
    threat_blocks: object = None              # list of (start, end): threat active only inside these blocks


@dataclass
class GridWorld:
    layout: Layout
    cfg: EnvConfig = field(default_factory=EnvConfig)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self.size = self.layout.size
        self.pos = np.array(self.layout.shelter)
        self.food_timer = {f: 0 for f in self.layout.food}
        self.threat_set = set(self.layout.threat)
        self.t = 0

    # ------------------------------------------------------------------
    def set_layout(self, layout: Layout) -> None:
        """Switch to a new world (the agent keeps its position)."""
        self.layout = layout
        self.food_timer = {f: 0 for f in layout.food}
        self.threat_set = set(layout.threat)

    def state(self) -> int:
        return int(self.pos[0] * self.size + self.pos[1])

    @property
    def n_states(self) -> int:
        return self.size * self.size

    def step(self, action: int) -> dict:
        new = np.clip(self.pos + MOVES[action], 0, self.size - 1)
        self.pos = new
        cell = (int(new[0]), int(new[1]))

        ate = False
        if cell in self.food_timer and self.food_timer[cell] == 0:
            ate = True
            self.food_timer[cell] = self.cfg.regrow_steps
        for f in self.food_timer:
            if self.food_timer[f] > 0 and not (ate and f == cell):
                self.food_timer[f] -= 1

        c = self.cfg
        active = c.threat_blocks is None or any(a <= self.t < b for a, b in c.threat_blocks)
        u = self.rng.random()   # always draw, so random streams stay aligned across conditions
        if c.no_damage or not active:
            damage = False
        elif c.damage_schedule is not None:
            damage = bool(c.damage_schedule[self.t]) if self.t < len(c.damage_schedule) else False
        elif c.uncontrollable:
            damage = u < c.p_damage_uncontrollable
        else:
            damage = cell in self.threat_set and u < c.p_damage_zone
        self.t += 1

        return {
            "ate": ate,
            "damage": bool(damage),
            "in_shelter": cell == tuple(self.layout.shelter),
            "social": self.layout.companion is not None and cell == tuple(self.layout.companion),
            "moved": action != 0,
        }
