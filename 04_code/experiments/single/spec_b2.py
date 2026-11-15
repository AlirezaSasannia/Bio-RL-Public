"""Pre-registration of stage B2 (hormones as an add-on module)."""
SPEC_B2 = {
    "id": "single_stageB2_addon",
    "registered": ("while stage B optimisation was running: stage-B training curves had been seen (high-dimensional "
                   "hormone controllers trained poorly under the equal budget), stage-B held-out tests had NOT"),
    "question": ("Equal-budget search from scratch disadvantages controllers with many parameters. Do hormones help as an "
                 "add-on to an already tuned learner? Same extra budget for the alternative of tuning the base further."),
    "bases": "the optimised 'fixed' and 'vdbe' controllers of each stage-B restart (4 x 2)",
    "arm_hormones": ("freeze the base; add the fuzzy endocrine module (NE, cortisol via MR/GR, ghrelin) to alpha, eps and "
                     "safety weight (for vdbe, hormones shift the per-state exploration in logit space); optimise only the "
                     "16 hormone parameters (4 membership centres, 12 gains), CMA-ES starting at zero gains, sigma 0.15, "
                     "20 generations x 12 candidates x 48 agents"),
    "arm_control": "continue CMA-ES on the base parameters alone from the base solution, sigma 0.15, same 20 x 12 x 48",
    "test": "stage-B held-out schedules (64 x 8 agents), schedule means averaged over restarts",
    "predictions": {"A1": "fixed + hormones > fixed tuned further (one-sided Wilcoxon over schedules)",
                    "A2": "vdbe + hormones > vdbe tuned further",
                    "family": "Holm across A1, A2"},
}
