# rulebook eval — bench-coder-60m-pass1

runs 1, game-runs 25, score mean 0.809, level1 rate 0.36, level2 rate 0.0, prediction ok/unknown/mismatch 0.43/0.4/0.15

Failure taxonomy (last level played): budget_time/actions_budget 9, mismatch_heavy 5, game_over_loop 5, predictor_blind 4, hypotheses_refuted 1, plans_ignored 1

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| r11l | 1 | 4.76±0.0 | 1/1 | predictor_blind |
| m0r0 | 1 | 3.94±0.0 | 1/1 | budget_time/actions_budget |
| tn36 | 1 | 3.57±0.0 | 1/1 | game_over_loop |
| lp85 | 1 | 2.78±0.0 | 1/1 | budget_time/actions_budget |
| sb26 | 1 | 2.78±0.0 | 1/1 | predictor_blind |
| ls20 | 1 | 1.33±0.0 | 1/1 | game_over_loop |
| ar25 | 1 | 0.67±0.0 | 1/1 | mismatch_heavy |
| su15 | 1 | 0.32±0.0 | 1/1 | game_over_loop |
| bp35 | 1 | 0.07±0.0 | 1/1 | hypotheses_refuted |
| cd82 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| cn04 | 1 | 0±0.0 | 0/0 | game_over_loop |
| dc22 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ft09 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| g50t | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ka59 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| lf52 | 1 | 0±0.0 | 0/0 | predictor_blind |
| re86 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| s5i5 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sc25 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| sk48 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| sp80 | 1 | 0±0.0 | 0/0 | game_over_loop |
| tr87 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| tu93 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| vc33 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| wa30 | 1 | 0±0.0 | 0/0 | plans_ignored |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260927-080029 | ar25 | 1 | 0.673 | [65] | [0.24] | 0.55/0.15/0.28 | 0 | 0/0 | 3/53 | 0 | 13 | 1 | mismatch_heavy |  (0.0) |
| 20260927-080029 | bp35 | 1 | 0.068 | [55] | [0.15] | 0.2/0.61/0.19 | 0 | 0/0 | 0/34 | 0 | 15 | 3 | hypotheses_refuted |  (0.0) |
| 20260927-080029 | cd82 | 0 | 0 | [] | [] | 0.29/0.19/0.52 | 0 | 0/0 | 0/141 | 0 | 9 | 1 | mismatch_heavy |  (0.0) |
| 20260927-080029 | cn04 | 0 | 0 | [] | [] | 0.39/0.4/0.16 | 0 | 0/0 | 0/9 | 0 | 11 | 4 | game_over_loop |  (0.0) |
| 20260927-080029 | dc22 | 0 | 0 | [] | [] | 0.38/0.59/0.03 | 0 | 0/0 | 3/98 | 0 | 5 | 1 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | ft09 | 0 | 0 | [] | [] | 0.8/0.16/0.04 | 0 | 0/0 | 0/0 | 0 | 4 | 1 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | g50t | 0 | 0 | [] | [] | 0.86/0.07/0.04 | 0 | 0/0 | 1/159 | 0 | 7 | 1 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | ka59 | 0 | 0 | [] | [] | 0.52/0.4/0.08 | 0 | 0/0 | 2/72 | 0 | 8 | 1 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | lf52 | 0 | 0 | [] | [] | 0.03/0.94/0.02 | 0 | 0/0 | 0/89 | 0 | 3 | 1 | predictor_blind |  (0.0) |
| 20260927-080029 | lp85 | 1 | 2.778 | [10] | [1.0] | 0.5/0.48/0.03 | 0 | 0/0 | 0/0 | 0 | 4 | 1 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | ls20 | 1 | 1.334 | [36] | [0.37] | 0.38/0.21/0.06 | 0 | 0/0 | 8/62 | 0 | 16 | 4 | game_over_loop |  (0.0) |
| 20260927-080029 | m0r0 | 1 | 3.935 | [33] | [0.83] | 0.91/0.05/0.03 | 0 | 0/0 | 2/3 | 0 | 5 | 0 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | r11l | 1 | 4.762 | [22] | [1.0] | 0.21/0.72/0.06 | 0 | 0/0 | 0/0 | 0 | 10 | 2 | predictor_blind |  (0.0) |
| 20260927-080029 | re86 | 0 | 0 | [] | [] | 0.65/0.05/0.29 | 0 | 0/0 | 9/20 | 0 | 10 | 2 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | s5i5 | 0 | 0 | [] | [] | 0.16/0.73/0.1 | 0 | 0/0 | 0/0 | 0 | 5 | 0 | predictor_blind |  (0.0) |
| 20260927-080029 | sb26 | 1 | 2.778 | [10] | [1.0] | 0.2/0.76/0.04 | 0 | 0/0 | 0/0 | 0 | 3 | 0 | predictor_blind |  (0.0) |
| 20260927-080029 | sc25 | 0 | 0 | [] | [] | 0.43/0.22/0.36 | 0 | 0/0 | 1/97 | 0 | 10 | 2 | mismatch_heavy |  (0.0) |
| 20260927-080029 | sk48 | 0 | 0 | [] | [] | 0.02/0.54/0.45 | 0 | 0/0 | 0/2 | 0 | 8 | 0 | mismatch_heavy |  (0.0) |
| 20260927-080029 | sp80 | 0 | 0 | [] | [] | 0.04/0.96/0.0 | 0 | 0/0 | 1/164 | 0 | 14 | 13 | game_over_loop |  (0.0) |
| 20260927-080029 | su15 | 1 | 0.32 | [22] | [1.0] | 0.17/0.78/0.05 | 0 | 0/0 | 0/0 | 0 | 12 | 4 | game_over_loop |  (0.0) |
| 20260927-080029 | tn36 | 1 | 3.571 | [9] | [1.0] | 0.96/0.03/0.02 | 0 | 0/0 | 0/0 | 0 | 5 | 3 | game_over_loop |  (0.0) |
| 20260927-080029 | tr87 | 0 | 0 | [] | [] | 0.25/0.13/0.62 | 0 | 0/0 | 0/72 | 0 | 8 | 0 | mismatch_heavy |  (0.0) |
| 20260927-080029 | tu93 | 0 | 0 | [] | [] | 0.46/0.37/0.17 | 0 | 0/0 | 1/51 | 0 | 10 | 2 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | vc33 | 0 | 0 | [] | [] | 0.72/0.27/0.02 | 0 | 0/0 | 0/0 | 0 | 2 | 2 | budget_time/actions_budget |  (0.0) |
| 20260927-080029 | wa30 | 0 | 0 | [] | [] | 0.57/0.31/0.11 | 0 | 0/0 | 0/1 | 0 | 9 | 1 | plans_ignored |  (0.0) |