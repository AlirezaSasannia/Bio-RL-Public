#!/usr/bin/env bash
# Stage C4 chain (Linux server): base -> register_arms -> arms -> test -> report; resumable (re-run to resume).
# Precondition: stageE2b_0/gate_summary.csv says PASS; C4-0 registered and its yaml committed (see RUNBOOK_fa.md).
# Parallel arms on a big GPU: instead of the single 'arms' line, start e.g. three processes:
#   python -u stage_c4.py arms gru,gru_aug & python -u stage_c4.py arms ff_big,film,generic & python -u stage_c4.py arms control,control_ra,aug,aug_ra,bias & wait
set -e
cd "$(dirname "$0")"
unset C4_SMOKE
L=../../results/logs/single
python -u stage_c4.py base          >> $L/stageC4_base.log   2>> $L/stageC4_base.err
python -u stage_c4.py register_arms >> $L/stageC4_arms.log   2>> $L/stageC4_arms.err
python -u stage_c4.py arms          >> $L/stageC4_arms.log   2>> $L/stageC4_arms.err
python -u stage_c4.py test          >> $L/stageC4_test.log   2>> $L/stageC4_test.err
python -u stage_c4.py report        >> $L/stageC4_report.log 2>> $L/stageC4_report.err
echo "C4 chain finished" >> $L/stageC4_arms.log
