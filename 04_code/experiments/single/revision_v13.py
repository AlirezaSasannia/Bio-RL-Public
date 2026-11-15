"""Amendment family revision_v1.3 (registered after the two simulated reviews of paper v1.2, before any revision run).
One spec for all three experiments (X1 stage_x1.py, X2/X3 stage_x2x3.py) and the planned re-analyses of existing logs.
Both scripts call register(SPEC) before their first evaluation; the hash must not change."""

SPEC = {
    "id": "single_revision_v1_3",
    "registered": ("2026-11-05, after the simulated reviews of v1.2 (07_paper/reviews/), before any revision run; code "
                   "smoke-tested on development seeds only (scratch output, nothing registered)"),
    "family": ("one amendment family. Primary family (Holm over all 7 primary tests together): X1-P1..P6 and X2-P1. "
               "Secondary family (Holm over all secondary tests together): X1-S1..S7, X2-S1..S3. X3 is evaluation-only "
               "pooling (own Holm, descriptive). All tests two-sided: no direction was predicted in advance"),
    "X1": {
        "question": ("Does the tabular global layer's gain over constant offsets need hidden regime switches, or does it "
                     "also appear in stationary single-regime lives (need/state-dependent control)?"),
        "agents": ("frozen B10 solutions (stageB10_opt/{control,bias,generic}_r{k}.json, k = 0..7; no re-tuning): "
                   "control = further-tuned VDBE base, bias = 3 constant offsets, generic = 15-weight global layer"),
        "conditions": ("c = 0..4: stationary life, safe juvenile steps 0-499 (as in every training life), then regime "
                       "R.NAMES[c] (safe, danger, uncontrollable, scarce, volatile) for steps 500-5999 (volatile keeps "
                       "its layout change every 300 steps); c = 5: switching reference, standard R.schedule lives from "
                       "new schedule seeds 53000-53127"),
        "test": ("new, never used: 128 groups x 4 agents per base, arm and condition; layout bank 5200000-5200999, "
                 "layout of agent j in group i = (37 j + 11 i) mod 1000; numpy simulator seed 320000 + 10 k + c, "
                 "shared by the three arms (common random numbers)"),
        "metric": ("score_post = mean over steps 500-5999 of wellbeing x alive (dead = 0), per agent, averaged over the "
                   "4 agents of a group; pooled stationary = mean over c = 0..4 of the group scores (same group index)"),
        "primary": {"X1-P1..P5": "generic vs bias in stationary condition c = 0..4 (two-sided)",
                    "X1-P6": "generic vs bias, pooled over the five stationary conditions (two-sided)"},
        "secondary": {"X1-S1..S5": "generic vs control in stationary condition c = 0..4 (two-sided)",
                      "X1-S6": "bias vs control, pooled stationary (two-sided)",
                      "X1-S7": ("(generic - bias) in switching lives minus (generic - bias) pooled stationary, per "
                                "base and group index (two-sided): does the gain over offsets grow with switches?")},
        "behaviour": ("descriptive only, all conditions, all 8 bases, read through the read-only simulate(hook=): per "
                      "regime and arm, while alive: move rate (action != stay), share of steps in the shelter, "
                      "Manhattan distance to the nearest food cell and to the nearest ready food cell, energy, the "
                      "three knob offsets and realised knobs (alpha, exploration, safety multiplier); time course in "
                      "100-step bins of knobs and behaviour in each stationary condition; conditional hazards (deaths "
                      "per 1000 agent-steps alive in the regime) with base-level t intervals"),
        "interpretation": ("If X1-P6 or any X1-P1..P5 passes with generic > bias, the gain over offsets does not need "
                           "regime switches: the result is reframed as need/state-dependent control, and X1-S7 says "
                           "whether switches add to it. If none passes, there is no detectable gain without switches "
                           "(not equivalence); if X1-S7 passes positive, the gain is larger with switches. A pass with "
                           "bias > generic is reported as such. Behaviour data are descriptive (no test)."),
    },
    "X2": {
        "question": ("Is the GRU's lead over the capacity-matched feed-forward net (C3c Q1) due to recurrence, or to its "
                     "sequence replay sampling?"),
        "arm": ("ff_big_seq = C3c ff_big (dqnv, last felt reward and action, hidden 113, 17,186 weights) with the GRU's "
                "replay sampling: each update 2 contiguous 4-step sequences (newest included, start ages uniform) "
                "instead of 8 single transitions; world.py cfg seq=True (all other learners bit-identical, checked)"),
        "budget": ("C3c budget on the 8 frozen C3 bases: CMA-ES 20 x 12 x 48, sigma 0.15, start at the C3 base, CMA "
                   "seed 28300 + k, generation seeds 61000000 + 1000 k + g and simulator 61000000 + 10 g + c (same as "
                   "C3/C3b/C3c: common random numbers)"),
        "test": ("new, never used: schedules 54000-54127 x 4 agents per base, layout bank 5300000-5300999, layouts "
                 "(37 j + 11 i) mod 1000, torch simulator seed 330000 + c (c = 0 bases 0-3, 1 bases 4-7); arms "
                 "ff_big_seq, gru (C3), ff_big (C3c), control_ra (C3b), frozen"),
        "primary": {"X2-P1": "gru vs ff_big_seq (two-sided)"},
        "secondary": {"X2-S1": "ff_big_seq vs ff_big (two-sided)", "X2-S2": "ff_big_seq vs control_ra (two-sided)",
                      "X2-S3": "gru vs ff_big (two-sided; re-test of C3c Q1)"},
        "interpretation": ("X2-P1 pass with gru > ff_big_seq: the lead remains with capacity and replay sampling "
                           "matched (recurrence; burn-in and other GRU details still differ). Pass with ff_big_seq > "
                           "gru: sequence sampling explains the lead. No pass: not established (not equivalence)."),
    },
    "X3": {
        "question": "How large are the C3/C3b/C3c effects when the three used test sets are pooled?",
        "design": ("evaluation only, no tuning: every frozen solution of control, aug, film, generic, bias, gru, "
                   "gru_aug, metagrad (C3), control_ra, aug_ra (C3b) and ff_big (C3c) on each of the three test sets "
                   "it lacks (C3 set: schedules 22000+, layouts 2200000+, simulator 110000 + c; C3b set: 44000+, "
                   "2400000+, 140000 + c; C3c set: 45000+, 2500000+, 150000 + c), with the seeds and protocol of that "
                   "set; existing evaluations are read from the logged raw CSVs. Determinism check: control is "
                   "re-evaluated on the C3b set and must equal the logged scores."),
        "analysis": ("for the comparisons of the C3, C3b and C3c registrations plus gru - ff_big and ff_big - aug_ra: "
                     "(i) base-level two-sided t-test on base means over all 384 schedules (n = 8, df = 7), t and "
                     "three-level bootstrap intervals; (ii) DerSimonian-Laird random-effects pooling of the three "
                     "per-set base-level estimates, between-set variance tau^2; (iii) schedule variance component "
                     "per set (base x schedule two-way decomposition). Holm within X3; descriptive: original "
                     "registered verdicts stand, disagreements are reported."),
    },
    "reanalyses": ("of logs already seen (no new test set): t-based (df = 7) intervals beside every bootstrap interval "
                   "of the independent-base tests; two-sided p for C3b/C3c; project-level table of every registration "
                   "with its outcome and a Benjamini-Hochberg sensitivity check over all registered primary "
                   "predictions; conditional hazards per regime for the B10 test set by deterministic re-computation "
                   "with its own seeds (scores must equal the logged ones); learned constant offsets of the B10 bias "
                   "arm; training fitness at CMA generations 0, 10 and 19 per arm"),
    "inference": ("unit = base (n = 8): two-sided one-sample t-test on base means of the paired difference, Holm "
                  "within family; 95% three-level bootstrap CI (reanalysis_runlevel.ci3, 4000 draws, seed 0) and 95% "
                  "t interval (df = 7). A claim needs p_Holm < 0.05 AND both intervals excluding 0. Exact Wilcoxon on "
                  "base means reported as a check. No detectable difference is not equivalence."),
}
