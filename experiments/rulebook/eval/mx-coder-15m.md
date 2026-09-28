# rulebook eval — mx-coder-15m

runs 1, game-runs 25, score mean 0.551, level1 rate 0.2, level2 rate 0.04, prediction ok/unknown/mismatch 0.43/0.43/0.1

Failure taxonomy (last level played): plans_ignored 5, predictor_blind 5, budget_time/actions_budget 5, game_over_loop 3, no_win_hypothesis 3, mismatch_heavy 3, hypotheses_refuted 1

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| sb26 | 1 | 6.42±0.0 | 2/2 | no_win_hypothesis |
| tn36 | 1 | 3.57±0.0 | 1/1 | game_over_loop |
| lp85 | 1 | 2.78±0.0 | 1/1 | predictor_blind |
| vc33 | 1 | 0.78±0.0 | 1/1 | predictor_blind |
| su15 | 1 | 0.23±0.0 | 1/1 | no_win_hypothesis |
| ar25 | 1 | 0±0.0 | 0/0 | plans_ignored |
| bp35 | 1 | 0±0.0 | 0/0 | plans_ignored |
| cd82 | 1 | 0±0.0 | 0/0 | predictor_blind |
| cn04 | 1 | 0±0.0 | 0/0 | game_over_loop |
| dc22 | 1 | 0±0.0 | 0/0 | plans_ignored |
| ft09 | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| g50t | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ka59 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |
| lf52 | 1 | 0±0.0 | 0/0 | predictor_blind |
| ls20 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| m0r0 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| r11l | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| re86 | 1 | 0±0.0 | 0/0 | plans_ignored |
| s5i5 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| sc25 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sk48 | 1 | 0±0.0 | 0/0 | plans_ignored |
| sp80 | 1 | 0±0.0 | 0/0 | game_over_loop |
| tr87 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| tu93 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| wa30 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260928-030547 | ar25 | 0 | 0 | [] | [] | 0.76/0.07/0.14 | 0 | 0/0 | 0/21 | 0 | 8 | 0 | plans_ignored |  (0.0) |
| 20260928-030547 | bp35 | 0 | 0 | [] | [] | 0.38/0.33/0.26 | 0 | 0/0 | 0/60 | 0 | 8 | 0 | plans_ignored |  (0.0) |
| 20260928-030547 | cd82 | 0 | 0 | [] | [] | 0.04/0.96/0.01 | 0 | 0/0 | 18/19 | 0 | 2 | 1 | predictor_blind |  (0.0) |
| 20260928-030547 | cn04 | 0 | 0 | [] | [] | 0.17/0.82/0.0 | 0 | 0/0 | 0/4 | 0 | 3 | 3 | game_over_loop |  (0.0) |
| 20260928-030547 | dc22 | 0 | 0 | [] | [] | 0.93/0.04/0.03 | 0 | 0/0 | 0/84 | 0 | 1 | 0 | plans_ignored |  (0.0) |
| 20260928-030547 | ft09 | 0 | 0 | [] | [] | 0.86/0.11/0.03 | 0 | 0/0 | 0/0 | 0 | 2 | 0 | no_win_hypothesis |  (0.0) |
| 20260928-030547 | g50t | 0 | 0 | [] | [] | 0.85/0.1/0.05 | 0 | 0/0 | 2/51 | 0 | 7 | 1 | budget_time/actions_budget |  (0.0) |
| 20260928-030547 | ka59 | 0 | 0 | [] | [] | 0.39/0.47/0.1 | 0 | 0/0 | 0/0 | 0 | 8 | 0 | hypotheses_refuted |  (0.0) |
| 20260928-030547 | lf52 | 0 | 0 | [] | [] | 0.0/1.0/0.0 | 0 | 0/0 | 1/32 | 0 | 1 | 1 | predictor_blind |  (0.0) |
| 20260928-030547 | lp85 | 1 | 2.778 | [15] | [1.0] | 0.12/0.88/0.0 | 0 | 0/0 | 1/31 | 0 | 1 | 0 | predictor_blind |  (0.0) |
| 20260928-030547 | ls20 | 0 | 0 | [] | [] | 0.49/0.13/0.33 | 0 | 0/0 | 0/23 | 0 | 8 | 0 | mismatch_heavy |  (0.0) |
| 20260928-030547 | m0r0 | 0 | 0 | [] | [] | 0.66/0.2/0.12 | 0 | 0/0 | 2/5 | 0 | 8 | 0 | budget_time/actions_budget |  (0.0) |
| 20260928-030547 | r11l | 0 | 0 | [] | [] | 0.42/0.48/0.1 | 0 | 0/0 | 0/0 | 0 | 6 | 1 | budget_time/actions_budget |  (0.0) |
| 20260928-030547 | re86 | 0 | 0 | [] | [] | 0.81/0.06/0.13 | 0 | 0/0 | 0/6 | 0 | 9 | 1 | plans_ignored |  (0.0) |
| 20260928-030547 | s5i5 | 0 | 0 | [] | [] | 0.37/0.53/0.1 | 0 | 0/0 | 0/0 | 0 | 5 | 0 | budget_time/actions_budget |  (0.0) |
| 20260928-030547 | sb26 | 2 | 6.425 | [11, 16] | [1.0, 1.0] | 0.37/0.6/0.02 | 0 | 0/0 | 0/0 | 0 | 5 | 1 | no_win_hypothesis |  (0.0) |
| 20260928-030547 | sc25 | 0 | 0 | [] | [] | 0.07/0.62/0.04 | 0 | 0/0 | 0/94 | 0 | 4 | 1 | predictor_blind |  (0.0) |
| 20260928-030547 | sk48 | 0 | 0 | [] | [] | 0.12/0.55/0.12 | 0 | 0/0 | 0/32 | 0 | 7 | 0 | plans_ignored |  (0.0) |
| 20260928-030547 | sp80 | 0 | 0 | [] | [] | 0.12/0.87/0.01 | 0 | 0/0 | 7/41 | 0 | 7 | 7 | game_over_loop |  (0.0) |
| 20260928-030547 | su15 | 1 | 0.226 | [22] | [1.0] | 0.65/0.28/0.06 | 0 | 0/0 | 0/0 | 0 | 5 | 1 | no_win_hypothesis |  (0.0) |
| 20260928-030547 | tn36 | 1 | 3.571 | [16] | [1.0] | 0.93/0.04/0.02 | 0 | 0/0 | 0/0 | 0 | 6 | 3 | game_over_loop |  (0.0) |
| 20260928-030547 | tr87 | 0 | 0 | [] | [] | 0.11/0.53/0.33 | 0 | 0/0 | 4/10 | 0 | 8 | 0 | mismatch_heavy |  (0.0) |
| 20260928-030547 | tu93 | 0 | 0 | [] | [] | 0.42/0.24/0.34 | 0 | 0/0 | 5/84 | 0 | 9 | 1 | mismatch_heavy |  (0.0) |
| 20260928-030547 | vc33 | 1 | 0.778 | [15] | [0.22] | 0.23/0.77/0.01 | 0 | 0/0 | 0/0 | 0 | 4 | 2 | predictor_blind |  (0.0) |
| 20260928-030547 | wa30 | 0 | 0 | [] | [] | 0.43/0.12/0.2 | 0 | 0/0 | 4/142 | 0 | 9 | 1 | budget_time/actions_budget |  (0.0) |