"""Resumable building blocks for stage scripts (added for stage B9).

cma_run     CMA-ES that checkpoints after every generation (optimiser object, history and the global numpy RNG state,
            which cma uses for sampling); an interrupted run continues exactly where it stopped, with the same result
            as an uninterrupted one.
cached_map  maps a job function over jobs with a pool; each finished job is written to its own file, so a restart only
            runs what is missing.
All writes are atomic (temp file + os.replace), so a kill during a write never leaves a corrupt file."""
import json
import os
import pickle
import time
from pathlib import Path

import cma
import numpy as np


def _atomic_write(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    for i in range(50):         # Windows: a reader holding the target open makes replace fail for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2)
    os.replace(tmp, path)


def cma_run(out_json: Path, x0, sigma, popsize, seed, gens, score_gen, make, meta: dict):
    """score_gen(X, g) -> array of scores (higher is better) for candidates X in generation g.
    make(u) -> parameter dict. Writes out_json with meta, params=make(es.mean), history; returns that dict."""
    out_json = Path(out_json)
    if out_json.exists():
        return json.loads(out_json.read_text())
    ckpt = out_json.with_suffix(".ckpt")
    if ckpt.exists():
        st = pickle.loads(ckpt.read_bytes())
        es, hist = st["es"], st["hist"]
        np.random.set_state(st["np_state"])
    else:
        es = cma.CMAEvolutionStrategy(np.asarray(x0, float), sigma,
                                      {"bounds": [0, 1], "popsize": popsize, "seed": seed, "verbose": -9})
        hist = []
    for g in range(len(hist), gens):
        X = es.ask()
        f = np.asarray(score_gen(X, g))
        es.tell(X, list(-f))
        hist.append(float(f.mean()))
        _atomic_write(ckpt, pickle.dumps(dict(es=es, hist=hist, np_state=np.random.get_state())))
    res = dict(**meta, params=make(es.mean), u=[float(v) for v in es.mean], history=hist)
    _atomic_write(out_json, json.dumps(res, indent=1).encode())
    ckpt.unlink(missing_ok=True)
    return res


def cached_map(pool, fn, jobs, keys, cache_dir: Path, log=print):
    """Run fn(job) for every job whose cache file cache_dir/<key>.pkl is missing; return results in job order."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = {k: cache_dir / f"{k}.pkl" for k in keys}
    todo = [(k, j) for k, j in zip(keys, jobs) if not path[k].exists()]
    if todo:
        log(f"  {len(todo)} of {len(jobs)} jobs to run ({len(jobs) - len(todo)} cached)")
        for k, res in pool.imap_unordered(_call, [(fn, k, j) for k, j in todo]):
            _atomic_write(path[k], pickle.dumps(res))
    return [pickle.loads(path[k].read_bytes()) for k in keys]


def _call(a):
    fn, k, j = a
    return k, fn(j)


def cma_lockstep(runs: list, gens: int, score_all, ckpt: Path, popsize: int, sigma: float):
    """Several CMA-ES runs advanced together, one generation at a time, so that all their candidates can be scored
    in one batched (GPU) call. runs: list of dicts with out_json, x0, seed, make (u -> params), meta.
    score_all(list of (run, X), g) -> list of score arrays. One checkpoint for all runs after every generation
    (optimisers, histories, global numpy RNG state). Finished runs write their out_json; returns their results."""
    ckpt = Path(ckpt)
    todo = [r for r in runs if not Path(r["out_json"]).exists()]
    if todo:
        if ckpt.exists():
            st = pickle.loads(ckpt.read_bytes())
            assert st["names"] == [str(r["out_json"]) for r in todo], "checkpoint belongs to a different set of runs"
            ess, hists = st["es"], st["hist"]
            np.random.set_state(st["np_state"])
        else:
            ess, hists = [], []
            for r in todo:
                ess.append(cma.CMAEvolutionStrategy(np.asarray(r["x0"], float), sigma,
                                                    {"bounds": [0, 1], "popsize": popsize, "seed": r["seed"],
                                                     "verbose": -9}))
                hists.append([])
        for g in range(len(hists[0]), gens):
            Xs = [es.ask() for es in ess]
            fs = score_all(list(zip(todo, Xs)), g)
            for es, X, f, h in zip(ess, Xs, fs, hists):
                es.tell(X, list(-np.asarray(f)))
                h.append(float(np.mean(f)))
            _atomic_write(ckpt, pickle.dumps(dict(names=[str(r["out_json"]) for r in todo], es=ess, hist=hists,
                                                  np_state=np.random.get_state())))
            print(f"  gen {g}: " + " ".join(f"{np.mean(f):.3f}" for f in fs), flush=True)
        for r, es, h in zip(todo, ess, hists):
            res = dict(**r["meta"], params=r["make"](es.mean), u=[float(v) for v in es.mean], history=h)
            _atomic_write(Path(r["out_json"]), json.dumps(res, indent=1).encode())
        ckpt.unlink(missing_ok=True)
    return [json.loads(Path(r["out_json"]).read_text()) for r in runs]
