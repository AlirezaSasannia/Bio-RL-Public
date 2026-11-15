"""Pre-registration of stage A (headroom) of the single-agent adaptation study."""
SPEC_A = {
    "id": "single_stageA_headroom",
    "question": ("In a world whose hidden regime changes within a life, how much better could an agent do if it "
                 "could set its learning parameters for the current regime (oracle) than the best agent with one "
                 "fixed setting? This bounds what ANY modulation scheme, biological or not, can gain."),
    "world": ("8x8, need-based agent without hormones (energy + safety; no belonging), 6000-step life, death after 100 "
              "steps at zero energy or as soon as safety reaches 0, safe juvenile period of 500 steps, then hidden "
              "regimes of 500-1200 steps in random order without immediate repeats. Regimes (src/bio_rl/fast/regimes.py): "
              "safe, danger (avoidable threat zone, p = 0.3), uncontrollable (random damage p = 0.006), scarce (food "
              "regrowth 60), volatile (world moves every 300 steps); lethality 0.4 per hit; metabolism 0.005."),
    "knobs": "learning rate alpha {0.1,0.3,0.6,0.9} x exploration eps {0.02,0.1,0.2,0.35,0.5} x safety-weight multiplier s {0.5,1,2,4} = 80 settings",
    "score": "wellbeing = mean over the 6000 steps of (1 - drive/2) while alive, 0 after death; survival reported too",
    "selection (training layouts 400000+, schedule seeds 0-15)": {
        "best_fixed": "argmax of score over the 80 settings in the switching world (16 schedules x 4 agents)",
        "oracle": "for each regime, argmax of score in that regime held constant for 3000 steps (64 agents); "
                  "the oracle agent switches to that setting whenever the regime changes (it is told the regime)",
    },
    "test (layouts 700000+, schedule seeds 1000-1063, 8 agents each, common random numbers)": {
        "headroom": "score(oracle) - score(best_fixed); 95% bootstrap CI over the 64 schedules",
        "relative_headroom": "(oracle - best_fixed) / (oracle - floor), floor = agent that never learns (alpha = 0)",
        "decision": ("go to stage B if the CI of the headroom excludes 0 AND relative headroom >= 0.10; otherwise the "
                     "world gives modulation too little to gain and must be redesigned (new registration)"),
        "also_reported": "headroom per regime (score accrued inside each regime), knob values chosen per regime",
    },
}
