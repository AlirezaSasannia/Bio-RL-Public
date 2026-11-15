"""Amendment family revision_v1.5 (registered after the three simulated reviews of paper v1.4, before any revision-v1.5
run). Evaluation only: frozen solutions, no new optimisation. One spec for X4 (truly switch-free lives), X5 (mechanism)
and X6 (one common test set); stage_x456.py calls register(SPEC) before its first evaluation; the hash must not change."""

SPEC = {
    "id": "single_revision_v1_5",
    "registered": ("2026-11-06, after the simulated reviews of v1.4 (07_paper/reviews/sim_review_v1.4_*.md), before any "
                   "v1.5 run; code smoke-tested on development seeds only (scratch output, nothing registered)"),
    "agents": ("frozen solutions only, no re-tuning. Tabular: stageB10_opt/{arm}_r{k}.json, k = 0..7 (eight independent "
               "VDBE bases; arms base, control = further tuning, bias = 3 constant offsets, generic = 15-weight global "
               "layer, yoked, augtab, auglin, bio = endocrine module) and stageB11_opt/generic_split_r{k}.json. Network "
               "(X6 only): the frozen C3/C3b/C3c solutions used in X3 (control, aug, film, generic, bias, gru, gru_aug, "
               "metagrad, control_ra, aug_ra, ff_big) and X2 ff_big_seq (revX2X3/ff_big_seq_opt)"),
    "family": ("one amendment family. Primary subfamily (Holm over X4-P1..P6). Secondary subfamily (Holm over all 18 "
               "secondary tests together: X4-S1..S8, X5-C1..C6, X5-D1..D4). One equivalence test X4-E1 (TOST, margin "
               "below). All superiority tests two-sided: no direction is predicted. X6 and every other X5 quantity are "
               "descriptive (no test)"),
    "X4": {
        "question": ("Does the global layer's gain over constant offsets appear in lives with no regime switch at all "
                     "(each life in one regime from step 0, no safe prelude)?"),
        "arms": "control, bias, generic (B10, frozen)",
        "conditions": ("c = 0..4: the whole life (steps 0-5999) in regime R.NAMES[c] (safe, danger, uncontrollable, "
                       "scarce, volatile); volatile keeps its layout change every 300 steps (that is the regime); usual "
                       "juvenile death protection (no death before step 500), no other change to the simulator"),
        "test": ("new, never used: 128 groups x 4 agents per base, arm and condition; layout bank 5400000-5400999, "
                 "layout of agent j in group i = (37 j + 11 i) mod 1000; numpy simulator seed 340000 + 10 k + c, shared "
                 "by all arms of base k (common random numbers)"),
        "metric": ("score = mean over steps 0-5999 of wellbeing x alive (dead = 0), per agent, averaged over the 4 "
                   "agents of a group; pooled = mean over c = 0..4 of the group scores (same group index). Descriptive: "
                   "score over steps 1000-5999 and in 500-step bins"),
        "primary": {"X4-P1..P5": "generic vs bias in condition c = 0..4 (two-sided)",
                    "X4-P6": "generic vs bias, pooled over the five conditions (two-sided)"},
        "secondary": {"X4-S1..S5": "generic vs control in condition c = 0..4 (two-sided)",
                      "X4-S6": "generic vs control, pooled (two-sided)",
                      "X4-S7": "bias vs control, pooled (two-sided)",
                      "X4-S8": ("(generic - bias) in the X5 switching lives (grace 100) minus (generic - bias) pooled "
                                "switch-free, per base and group index (two-sided)")},
        "equivalence": ("X4-E1: the same difference as X4-S8; TOST with margin +-0.025 wellbeing units (about half the "
                        "registered B10 gain over offsets, +0.054, and half the X1 switching gain, +0.047); equivalence "
                        "is claimed if the 90% t interval over the 8 base means (df 7) lies inside (-0.025, 0.025), "
                        "i.e. both one-sided t-tests p < 0.05"),
        "also_reported": ("descriptive, X1 lives (already used set, deterministic re-computation with the X1 seeds; "
                          "scores must equal revX1/x1_groups.csv): generic - bias and generic - control over steps "
                          "1000-5999 per stationary condition and pooled, base-level t intervals, no test"),
        "interpretation": ("X4-P6 or any X4-P1..P5 passing with generic > bias: the gain over offsets is present without "
                           "any regime switch (state-dependent control). No pass: not shown without switches (not "
                           "equivalence). X4-E1 passing: switches add less than +-0.025 to the gain; X4-S8 passing "
                           "positive: the gain is larger with switches. A pass with bias > generic is reported as such."),
    },
    "X5": {
        "question": ("What produces the survival gain: cause of death, time at zero energy, dependence on the death "
                     "rule, which knob carries the state dependence, and how strongly the signals co-vary?"),
        "lives": ("(a) the X4 lives; (b) new switching lives: standard R.schedule from schedule seeds 56000-56127, 4 "
                  "agents per schedule, layout bank 5500000-5500999, layouts (37 j + 11 i) mod 1000, numpy simulator "
                  "seed 350000 + 10 k + v (v = 0 grace 100, v = 1 grace 50, v = 2 grace 200), shared by all arms of "
                  "base k; arms control, bias, generic, bio (bio descriptive only)"),
        "death_rules": ("starvation death after G consecutive steps at zero energy, G = 100 (as trained), 50 and 200; "
                        "injury death at safety 0 unchanged; juvenile protection unchanged. The X4 lives under G = 50 and "
                        "200 use the X4 seeds (same worlds)"),
        "knob_clamps": ("generic with one output held constant: for knob j (alpha, exploration, safety weight) the 5 "
                        "weights of that knob are set to 0 and its constant offset to m_j(k) = mean of the realised "
                        "offset of knob j over all alive agent-steps of the unclamped generic arm of base k in the same "
                        "lives (computed in a first pass); the leak (tau 50) and the other two knobs are unchanged. In "
                        "X4 scarcity (seed of X4 c = 3) and X5 switching lives (G = 100, seed v = 0): same worlds as the "
                        "reference"),
        "registered_contrasts": {"X5-C1..C3": "generic vs generic with alpha / exploration / safety clamped, X4 "
                                              "scarcity (two-sided)",
                                 "X5-C4..C6": "the same in X5 switching lives (two-sided)",
                                 "X5-D1, D2": "generic vs bias in X5 switching lives under G = 50, G = 200 (two-sided)",
                                 "X5-D3, D4": "generic vs bias pooled over the X4 conditions under G = 50, 200"},
        "descriptive": ("per regime and arm: deaths by cause (starvation: G consecutive zero-energy steps; injury: "
                        "safety 0) per 1000 agent-steps alive (cause-specific hazards) with base-level t intervals; share "
                        "of alive steps at zero energy; longest zero-energy run; Pearson correlations among the five "
                        "appraisal inputs (threat, spatial threat predictability K, surprise, intake, energy deficit) "
                        "over alive steps per regime and base (generic and bias arms), mean and t interval over bases; "
                        "the same quantities for the B10 test set by deterministic re-computation with its own seeds "
                        "(scores must equal the log), plus Kaplan-Meier curves and restricted mean survival time "
                        "(tau = 6000) per arm with the base as cluster"),
        "interpretation": ("A clamp that lowers the score (pass, generic > clamp) shows that the state dependence of "
                           "that knob is needed in those lives; no pass is not evidence that it is not needed. X5-D1..D4 "
                           "passing: the gain over offsets is not tied to the trained death rule G = 100 (frozen agents, "
                           "not re-tuned). Causes and correlations are descriptive."),
    },
    "X6": {
        "question": "Absolute scores of all frozen eight-base solutions on one common test set (descriptive).",
        "test": ("new, never used: 4096 schedules (seeds 65000-69095), layout bank 5600000-5600999, layout of agent j "
                 "in schedule i = (37 j + 11 i) mod 1000. Tabular: 4 agents per schedule, all arms listed under "
                 "agents (B10 and B11), numpy simulator seed 360000 + 100 k + chunk (chunks of 256 schedules), shared "
                 "by all arms of base k. Network (GPU): schedules 65000-66023, agent j = 0 only (1024 agents per base; "
                 "the full 4096-schedule set would take about 15 h on the RTX 2050, estimated from the X3 timings, "
                 "above the 6 h limit), torch simulator seed 370000 + c (c = 0..3, two bases per call)"),
        "analysis": ("per arm: mean score, survival and alive wellbeing with t intervals over the 8 bases; the "
                     "registered pairwise comparisons of B10, B11, C3, C3b, C3c, X2 recomputed on this set (base-level "
                     "t interval, ci3, two-sided p, Holm within X6); descriptive only: registered verdicts unchanged"),
    },
    "inference": ("unit = base (n = 8): two-sided one-sample t-test on base means of the paired difference (differences "
                  "averaged over groups/schedules within base), Holm within the subfamily; 95% three-level bootstrap CI "
                  "(reanalysis_runlevel.ci3, 4000 draws, seed 0) and 95% t interval (df 7). A claim needs p_Holm < 0.05 "
                  "AND both intervals excluding 0. Also reported beside every t-test: the exact sign-flip permutation p "
                  "over the 8 base means (2^8 assignments, two-sided) and d_z = mean / sd of the base means. No "
                  "detectable difference is not equivalence; for every non-detection the report states which effects "
                  "the t interval excludes."),
}
