"""Run one agent through a training world, then switch it to a novel world."""
from __future__ import annotations

import numpy as np

from bio_rl.envs.gridworld import LAYOUT_A, LAYOUT_B, EnvConfig, GridWorld
from bio_rl.learning.agents import Agent, AgentConfig


def run(agent_cfg: AgentConfig, seed: int, train_steps: int = 4000, test_steps: int = 4000,
        env_cfg: EnvConfig | None = None, switch: bool = True) -> dict:
    """Return per-step traces. The world switches from layout A to B at ``train_steps``."""
    rng = np.random.default_rng(seed)
    env = GridWorld(LAYOUT_A, env_cfg or EnvConfig(), rng)
    agent = Agent(env.n_states, agent_cfg, rng)
    T = train_steps + test_steps
    out = {k: np.zeros(T) for k in ("energy", "safety", "drive", "ate", "damage", "NE", "C", "K", "eps", "w_safety_share", "in_shelter")}
    s = env.state()
    for t in range(T):
        if switch and t == train_steps:
            env.set_layout(LAYOUT_B)
        a = agent.act(s)
        ev = env.step(a)
        s2 = env.state()
        info = agent.learn(s, a, s2, ev)
        s = s2
        w = agent.weights()
        out["energy"][t], out["safety"][t] = agent.needs.n
        out["drive"][t] = agent.needs.drive()
        out["ate"][t] = ev["ate"]
        out["damage"][t] = ev["damage"]
        out["in_shelter"][t] = ev["in_shelter"]
        out["NE"][t], out["C"][t], out["K"][t] = info["NE"], info["C"], info["K"]
        out["eps"][t] = agent.epsilon()
        out["w_safety_share"][t] = w[-1] / w.sum() if len(w) > 1 else np.nan
    return out


def run_schedule(agent_cfg: AgentConfig, seed: int, schedule: list, total_steps: int,
                 env_cfg: EnvConfig | None = None) -> dict:
    """Run with a list of (step, Layout) world changes; the first entry must be at step 0."""
    rng = np.random.default_rng(seed)
    changes = dict(schedule)
    env = GridWorld(changes[0], env_cfg or EnvConfig(), rng)
    agent = Agent(env.n_states, agent_cfg, rng)
    keys = ("drive", "ate", "damage", "C", "NE", "OT", "alpha", "in_shelter", "social",
            "energy", "safety", "belonging")
    out = {k: np.zeros(total_steps) for k in keys}
    s = env.state()
    for t in range(total_steps):
        if t in changes and t > 0:
            env.set_layout(changes[t])
        a = agent.act(s)
        ev = env.step(a)
        s2 = env.state()
        info = agent.learn(s, a, s2, ev)
        s = s2
        out["drive"][t] = agent.needs.drive()
        out["ate"][t], out["damage"][t], out["in_shelter"][t] = ev["ate"], ev["damage"], ev["in_shelter"]
        out["C"][t], out["NE"][t], out["OT"][t] = info["C"], info["NE"], info.get("OT", 0.0)
        out["alpha"][t] = agent.alpha()
        out["social"][t] = ev.get("social", False)
        n = agent.needs.n
        out["energy"][t], out["safety"][t] = n[0], n[1]
        out["belonging"][t] = n[2] if len(n) > 2 else np.nan
    return out


def run_phases(agent_cfg: AgentConfig, seed: int, phases: list) -> dict:
    """Run one agent through consecutive phases [(steps, Layout, EnvConfig), ...].

    The agent (values, needs, hormones) carries over between phases; each phase
    gets a fresh environment clock so damage schedules and threat blocks are
    indexed from the start of that phase.
    """
    rng = np.random.default_rng(seed)
    total = sum(p[0] for p in phases)
    env = GridWorld(phases[0][1], phases[0][2], rng)
    agent = Agent(env.n_states, agent_cfg, rng)
    keys = ("drive", "ate", "damage", "C", "NE", "OT", "alpha", "in_shelter", "social",
            "energy", "safety", "belonging", "K", "Kp", "moved")
    out = {k: np.zeros(total) for k in keys}
    s, t = env.state(), 0
    for steps, layout, cfg in phases:
        env.set_layout(layout)
        env.cfg, env.t = cfg, 0
        for _ in range(steps):
            a = agent.act(s)
            ev = env.step(a)
            s2 = env.state()
            info = agent.learn(s, a, s2, ev)
            s = s2
            out["drive"][t] = agent.needs.drive()
            out["ate"][t], out["damage"][t], out["in_shelter"][t] = ev["ate"], ev["damage"], ev["in_shelter"]
            out["C"][t], out["NE"][t], out["OT"][t], out["K"][t] = info["C"], info["NE"], info.get("OT", 0.0), info["K"]
            out["Kp"][t], out["moved"][t] = info.get("Kp", 0.5), ev["moved"]
            out["alpha"][t] = agent.alpha()
            out["social"][t] = ev.get("social", False)
            n = agent.needs.n
            out["energy"][t], out["safety"][t] = n[0], n[1]
            out["belonging"][t] = n[2] if len(n) > 2 else np.nan
            t += 1
    return out
