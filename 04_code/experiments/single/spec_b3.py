"""Pre-registration of stage B3: placebo controls for the hormone add-on (reviewer P0/P1)."""
SPEC_B3 = {
    "id": "single_stageB3_placebo",
    "registered": "after an external review of draft v0.2 and before any B3 run",
    "question": "Does the add-on gain come from the biological rules, from slow hormonal dynamics, or from any extra layer?",
    "base": "the 4 optimised VDBE controllers of stage B, frozen; 2 add-on optimisations per base = 8 runs per arm",
    "arms": {
        "bio": "fuzzy endocrine module (literature rules, MR/GR, NE/cortisol/ghrelin dynamics), 16 parameters",
        "generic": ("generic linear meta-controller: the same 5 appraisal inputs (threat, controllability, surprise, "
                    "intake, energy deficit) mapped linearly to the same 3 knob offsets, with a leaky memory "
                    "(tau 50); 15 parameters"),
        "randrules": "the bio architecture with randomly drawn rules (fixed per run, seed 100 + run); 16 parameters",
        "nomem": "the bio module with instantaneous hormones (no hormonal time constants); 16 parameters",
        "control": "no module: the same extra budget spent on tuning the VDBE base further",
    },
    "optimisation": ("CMA-ES, 20 generations x 12 candidates x 48 agents, sigma 0.15, starting at zero effect; runs 0-3 "
                     "of bio and control are the stage-B2 runs (same protocol), runs 4-7 and all other arms are new"),
    "test": ("fresh: 128 new schedules (seeds 3000-3127) x 4 agents, layouts 800000+, common random numbers across arms; "
             "never used before"),
    "inference": ("for each comparison bio - X: (i) one-sided Wilcoxon over the 8 run-level mean differences "
                  "(runs paired by base and seed), Holm across C1-C4; (ii) hierarchical bootstrap CI (runs, then "
                  "schedules). A claim needs both (i) p_Holm < 0.05 and (ii) CI excluding 0"),
    "predictions": {"C1": "bio > generic", "C2": "bio > randrules", "C3": "bio > nomem", "C4": "bio > control"},
    "secondary": {
        "decomposition": "survival and mean wellbeing while alive, per arm",
        "per_regime": "difference bio - control inside each regime with CI",
        "ood": ("out-of-distribution test sets, no re-optimisation: short regimes (200-500 steps), long regimes "
                "(1500-3000), lethal hits (loss 0.6)"),
        "warm_start": ("phase 2: express each B2 fixed+hormones solution in the stage-B 'fuzzy' space and continue "
                       "joint CMA-ES (19 parameters, 30 generations, sigma 0.15). If the B failure was a search failure, "
                       "the result stays at or above the B2 solution and far above B's from-scratch fuzzy (-67%)"),
    },
}

AMENDMENT_B3 = {
    "id": "single_stageB3_amendment1",
    "amends": "single_stageB3_placebo",
    "when": "after 2 of 36 optimisation runs (training curves only), before any test evaluation",
    "change": ("4 runs per arm instead of 8 (one per VDBE base; bio and control = the stage-B2 runs). Primary "
               "inference: one-sided Wilcoxon over the 128 test schedules of the run-averaged difference, Holm "
               "across C1-C4, AND a hierarchical bootstrap CI (runs, then schedules) excluding 0. The run-level "
               "sign (4 of 4) is reported descriptively."),
    "reason": "compute: one run takes ~6.5 min per core, 36 runs would take ~2 h; the user asked for faster turnaround",
}
