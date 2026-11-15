"""Pre-registration of stage B of the single-agent adaptation study."""
SPEC_B = {
    "id": "single_stageB_controllers",
    "question": ("Without being told the hidden regime, how much of the stage-A headroom (oracle - best fixed) does each "
                 "way of setting the learner's knobs online capture, and do biologically inspired controllers beat "
                 "standard adaptive methods?"),
    "world": "as single_stageA_headroom (5 hidden regimes switching within a 6000-step life)",
    "simulator": "bio_rl.fast.adaptive (fixed controller equivalent to popsim: results/logs/single/validate_adaptive.csv)",
    "controllers": {
        "fixed": "constant alpha, eps, safety weight (3 parameters)",
        "reactive": "endocrine v0.5 channels on top of a fixed base (10)",
        "fuzzy": ("rules from endocrinology -> NE, cortisol, ghrelin; cortisol acts through MR and GR receptors; "
                  "hormones change alpha, eps and safety weight with evolved gains (19)"),
        "plastic": "fuzzy + receptor plasticity: within-life, reward-modulated three-factor learning of receptor sensitivity (21)",
        "vdbe": "value-difference based exploration, Tokic 2010 (4)",
        "changept": "Page-Hinkley change detection on surprise, boosts alpha and eps after a change (8)",
        "pearcehall": "surprise-driven adaptive learning rate (4)",
        "oracle": "stage-A per-regime knob table, told the regime (reference only, not optimised)",
    },
    "rules_source": ("fuzzy rules written from the endocrine literature before seeing any stage-B result and without "
                     "using the stage-A oracle table: unexpected surprise without threat -> NE (Yu & Dayan 2005); "
                     "threat with control -> moderate cortisol, threat without control -> high cortisol (Dickerson & "
                     "Kemeny 2004); low intake and low energy without threat -> ghrelin, suppressed by threat; cortisol "
                     "transduced by high-affinity MR and low-affinity GR receptors (de Kloet et al. 2005)"),
    "optimisation": ("CMA-ES on the unit cube, identical budget for every controller: 4 restarts x 30 generations x 12 "
                     "candidates x 48 agents (48 random schedules, training layouts 400000+); common random numbers "
                     "across candidates and controllers within a generation; solution = final CMA mean"),
    "test": ("64 new schedules (seeds 2000-2063) x 8 agents, layouts 700000+, identical layouts and random seeds for "
             "every controller; unit of inference = schedule mean (averaged over restarts and agents)"),
    "primary_measure": "score (wellbeing); headroom captured = (controller - fixed) / (oracle - fixed)",
    "predictions": {
        "H1": "plastic > vdbe, plastic > changept, plastic > pearcehall (one-sided Wilcoxon over schedules, Holm across 3)",
        "H2": "plastic > fuzzy (receptor plasticity adds adaptation)",
        "H3": "fuzzy > reactive (structured rules beat v0.5 reactive channels)",
        "H4": "plastic > fixed",
        "family_H2_H4": "Holm across H2-H4",
    },
    "secondary": {
        "lesions": ("in the optimised fuzzy and plastic agents, remove NE, cortisol or ghrelin effects (gains to 0): "
                    "expected losses in volatile (NE), danger/uncontrollable (cortisol) and scarce (ghrelin) periods"),
        "adaptation": "score in the first 300 steps after each regime switch, per controller",
        "per_regime": "headroom captured per regime",
    },
}
