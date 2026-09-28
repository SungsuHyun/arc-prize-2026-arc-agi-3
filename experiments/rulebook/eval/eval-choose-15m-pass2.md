# rulebook eval — eval-choose-15m-pass2

runs 1, game-runs 25, score mean 0.746, level1 rate 0.12, level2 rate 0.08, prediction ok/unknown/mismatch 0.42/0.42/0.11

Failure taxonomy (last level played): budget_time/actions_budget 7, stuck_repeating 6, no_win_hypothesis 4, hypotheses_refuted 4, predictor_blind 2, mismatch_heavy 1, game_over_loop 1

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| vc33 | 1 | 8.54±0.0 | 2/2 | no_win_hypothesis |
| lp85 | 1 | 6.54±0.0 | 2/2 | predictor_blind |
| tn36 | 1 | 3.57±0.0 | 1/1 | budget_time/actions_budget |
| ar25 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| bp35 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| cd82 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| cn04 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| dc22 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| ft09 | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| g50t | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ka59 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |
| lf52 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ls20 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |
| m0r0 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| r11l | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| re86 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| s5i5 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| sb26 | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| sc25 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| sk48 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sp80 | 1 | 0±0.0 | 0/0 | game_over_loop |
| su15 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |
| tr87 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| tu93 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| wa30 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260927-233151 | ar25 | 0 | 0 | [] | [] | 0.84/0.07/0.06 | 45 | 0/0 | 1/43 | 0 | 3 | 1 | stuck_repeating | DOWN (0.42) |
| 20260927-233151 | bp35 | 0 | 0 | [] | [] | 0.25/0.33/0.33 | 24 | 1/1 | 1/11 | 0 | 7 | 1 | stuck_repeating | RIGHT (0.5) |
| 20260927-233151 | cd82 | 0 | 0 | [] | [] | 0.0/0.99/0.01 | 166 | 0/6 | 91/91 | 0 | 1 | 0 | stuck_repeating | plan:collect(W2) (0.55) |
| 20260927-233151 | cn04 | 0 | 0 | [] | [] | 0.53/0.06/0.41 | 62 | 0/2 | 3/28 | 0 | 9 | 1 | mismatch_heavy | DOWN (0.31) |
| 20260927-233151 | dc22 | 0 | 0 | [] | [] | 0.9/0.06/0.03 | 63 | 0/0 | 0/62 | 0 | 2 | 0 | stuck_repeating | UP (0.4) |
| 20260927-233151 | ft09 | 0 | 0 | [] | [] | 0.85/0.11/0.04 | 25 | 0/0 | 0/0 | 0 | 1 | 0 | no_win_hypothesis | click(9@38,38) (0.12) |
| 20260927-233151 | g50t | 0 | 0 | [] | [] | 0.88/0.06/0.06 | 54 | 0/1 | 3/53 | 0 | 7 | 0 | budget_time/actions_budget | UP (0.33) |
| 20260927-233151 | ka59 | 0 | 0 | [] | [] | 0.17/0.29/0.36 | 53 | 0/5 | 0/0 | 0 | 8 | 0 | hypotheses_refuted | DOWN (0.36) |
| 20260927-233151 | lf52 | 0 | 0 | [] | [] | 0.07/0.6/0.02 | 104 | 15/4 | 6/139 | 0 | 4 | 1 | budget_time/actions_budget | click(14@19,18) (0.13) |
| 20260927-233151 | lp85 | 2 | 6.536 | [9, 49] | [1.0, 0.6] | 0.1/0.9/0.0 | 58 | 0/1 | 4/22 | 0 | 2 | 0 | predictor_blind | plan:learn(W5) (0.5) |
| 20260927-233151 | ls20 | 0 | 0 | [] | [] | 0.51/0.04/0.26 | 42 | 0/0 | 0/0 | 0 | 8 | 0 | hypotheses_refuted | DOWN (0.24) |
| 20260927-233151 | m0r0 | 0 | 0 | [] | [] | 0.75/0.05/0.16 | 18 | 0/0 | 1/3 | 0 | 6 | 0 | budget_time/actions_budget | DOWN (0.39) |
| 20260927-233151 | r11l | 0 | 0 | [] | [] | 0.1/0.76/0.14 | 71 | 0/3 | 0/0 | 0 | 8 | 1 | no_win_hypothesis | click(6@57,27) (0.13) |
| 20260927-233151 | re86 | 0 | 0 | [] | [] | 0.93/0.02/0.05 | 47 | 0/0 | 14/70 | 0 | 6 | 1 | budget_time/actions_budget | DOWN (0.3) |
| 20260927-233151 | s5i5 | 0 | 0 | [] | [] | 0.0/0.9/0.1 | 50 | 0/4 | 0/0 | 0 | 5 | 0 | stuck_repeating | click(14@20,38) (0.48) |
| 20260927-233151 | sb26 | 0 | 0 | [] | [] | 0.08/0.86/0.06 | 36 | 0/2 | 0/0 | 0 | 2 | 0 | no_win_hypothesis | SPACE (0.36) |
| 20260927-233151 | sc25 | 0 | 0 | [] | [] | 0.03/0.52/0.08 | 60 | 0/2 | 7/97 | 0 | 6 | 1 | budget_time/actions_budget | LEFT (0.33) |
| 20260927-233151 | sk48 | 0 | 0 | [] | [] | 0.1/0.84/0.05 | 51 | 0/2 | 0/2 | 0 | 4 | 0 | predictor_blind | DOWN (0.29) |
| 20260927-233151 | sp80 | 0 | 0 | [] | [] | 0.06/0.93/0.01 | 87 | 2/40 | 8/75 | 0 | 8 | 6 | game_over_loop | DOWN (0.29) |
| 20260927-233151 | su15 | 0 | 0 | [] | [] | 0.04/0.96/0.0 | 106 | 7/0 | 0/0 | 0 | 0 | 0 | hypotheses_refuted | ACTION7 (0.92) |
| 20260927-233151 | tn36 | 1 | 3.571 | [10] | [1.0] | 0.76/0.19/0.05 | 74 | 7/3 | 0/0 | 1 | 5 | 1 | budget_time/actions_budget | click(1@33,39) (0.05) |
| 20260927-233151 | tr87 | 0 | 0 | [] | [] | 0.99/0.01/0.0 | 104 | 0/0 | 103/206 | 0 | 0 | 0 | stuck_repeating | plan:collect(W1) (0.99) |
| 20260927-233151 | tu93 | 0 | 0 | [] | [] | 0.72/0.07/0.21 | 28 | 0/0 | 2/27 | 0 | 8 | 0 | budget_time/actions_budget | RIGHT (0.36) |
| 20260927-233151 | vc33 | 2 | 8.545 | [23, 11] | [0.09, 1.0] | 0.26/0.74/0.0 | 34 | 1/0 | 0/0 | 0 | 2 | 0 | no_win_hypothesis |  (0.0) |
| 20260927-233151 | wa30 | 0 | 0 | [] | [] | 0.48/0.08/0.33 | 92 | 0/1 | 0/0 | 0 | 8 | 0 | hypotheses_refuted | DOWN (0.4) |