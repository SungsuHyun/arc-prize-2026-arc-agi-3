# rulebook eval — eval-choose-15m-pass1

runs 1, game-runs 25, score mean 0.674, level1 rate 0.08, level2 rate 0.04, prediction ok/unknown/mismatch 0.46/0.33/0.13

Failure taxonomy (last level played): budget_time/actions_budget 10, stuck_repeating 4, predictor_blind 3, game_over_loop 2, plans_ignored 2, mismatch_heavy 2, no_win_hypothesis 1, hypotheses_refuted 1

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| lp85 | 1 | 16.67±0.0 | 3/3 | no_win_hypothesis |
| vc33 | 1 | 0.19±0.0 | 1/1 | stuck_repeating |
| ar25 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| bp35 | 1 | 0±0.0 | 0/0 | game_over_loop |
| cd82 | 1 | 0±0.0 | 0/0 | predictor_blind |
| cn04 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| dc22 | 1 | 0±0.0 | 0/0 | plans_ignored |
| ft09 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| g50t | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ka59 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| lf52 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| ls20 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| m0r0 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| r11l | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| re86 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| s5i5 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sb26 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sc25 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| sk48 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| sp80 | 1 | 0±0.0 | 0/0 | game_over_loop |
| su15 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| tn36 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| tr87 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| tu93 | 1 | 0±0.0 | 0/0 | plans_ignored |
| wa30 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260927-124916 | ar25 | 0 | 0 | [] | [] | 0.71/0.08/0.21 | 11 | 0/0 | 1/13 | 0 | 4 | 0 | budget_time/actions_budget | LEFT (0.36) |
| 20260927-124916 | bp35 | 0 | 0 | [] | [] | 0.23/0.3/0.35 | 68 | 0/1 | 0/57 | 0 | 11 | 3 | game_over_loop | RIGHT (0.41) |
| 20260927-124916 | cd82 | 0 | 0 | [] | [] | 0.0/0.99/0.01 | 119 | 2/2 | 13/26 | 0 | 2 | 1 | predictor_blind | UP (0.3) |
| 20260927-124916 | cn04 | 0 | 0 | [] | [] | 0.36/0.37/0.26 | 66 | 0/0 | 1/12 | 0 | 9 | 1 | budget_time/actions_budget | DOWN (0.27) |
| 20260927-124916 | dc22 | 0 | 0 | [] | [] | 0.88/0.07/0.05 | 40 | 0/0 | 0/54 | 0 | 2 | 0 | plans_ignored | RIGHT (0.33) |
| 20260927-124916 | ft09 | 0 | 0 | [] | [] | 0.88/0.09/0.02 | 86 | 61/0 | 0/0 | 0 | 2 | 0 | budget_time/actions_budget | click(9@38,38) (0.08) |
| 20260927-124916 | g50t | 0 | 0 | [] | [] | 0.92/0.03/0.03 | 91 | 0/1 | 3/88 | 0 | 6 | 1 | budget_time/actions_budget | UP (0.29) |
| 20260927-124916 | ka59 | 0 | 0 | [] | [] | 0.18/0.25/0.33 | 51 | 2/12 | 0/50 | 0 | 8 | 0 | mismatch_heavy | DOWN (0.31) |
| 20260927-124916 | lf52 | 0 | 0 | [] | [] | 0.04/0.37/0.02 | 102 | 0/3 | 65/129 | 0 | 3 | 1 | stuck_repeating | plan:collect(W3) (0.64) |
| 20260927-124916 | lp85 | 3 | 16.667 | [12, 22, 30] | [1.0, 1.0, 1.0] | 0.47/0.53/0.0 | 40 | 0/0 | 12/28 | 0 | 3 | 0 | no_win_hypothesis |  (0.0) |
| 20260927-124916 | ls20 | 0 | 0 | [] | [] | 0.15/0.07/0.18 | 66 | 0/0 | 1/5 | 0 | 8 | 0 | budget_time/actions_budget | RIGHT (0.21) |
| 20260927-124916 | m0r0 | 0 | 0 | [] | [] | 0.74/0.07/0.15 | 20 | 0/1 | 1/2 | 0 | 7 | 0 | budget_time/actions_budget | DOWN (0.35) |
| 20260927-124916 | r11l | 0 | 0 | [] | [] | 0.46/0.5/0.04 | 96 | 1/1 | 0/0 | 0 | 5 | 1 | budget_time/actions_budget | click(15@18,38) (0.17) |
| 20260927-124916 | re86 | 0 | 0 | [] | [] | 0.84/0.04/0.05 | 31 | 0/0 | 2/9 | 0 | 3 | 0 | stuck_repeating | DOWN (0.45) |
| 20260927-124916 | s5i5 | 0 | 0 | [] | [] | 0.02/0.89/0.09 | 47 | 4/0 | 0/0 | 0 | 4 | 0 | predictor_blind | click(4@38,24) (0.23) |
| 20260927-124916 | sb26 | 0 | 0 | [] | [] | 0.24/0.65/0.1 | 49 | 12/0 | 0/0 | 0 | 5 | 0 | predictor_blind | SPACE (0.29) |
| 20260927-124916 | sc25 | 0 | 0 | [] | [] | 0.14/0.49/0.23 | 35 | 0/4 | 4/84 | 0 | 8 | 0 | budget_time/actions_budget | DOWN (0.37) |
| 20260927-124916 | sk48 | 0 | 0 | [] | [] | 0.37/0.27/0.36 | 56 | 0/3 | 0/1 | 0 | 8 | 0 | mismatch_heavy | DOWN (0.25) |
| 20260927-124916 | sp80 | 0 | 0 | [] | [] | 0.39/0.6/0.01 | 73 | 1/17 | 6/64 | 0 | 8 | 6 | game_over_loop | DOWN (0.27) |
| 20260927-124916 | su15 | 0 | 0 | [] | [] | 0.53/0.45/0.02 | 83 | 3/2 | 0/0 | 0 | 3 | 2 | budget_time/actions_budget | ACTION7 (0.17) |
| 20260927-124916 | tn36 | 0 | 0 | [] | [] | 0.83/0.14/0.03 | 123 | 2/18 | 0/0 | 0 | 4 | 2 | budget_time/actions_budget | click(4@16,31) (0.09) |
| 20260927-124916 | tr87 | 0 | 0 | [] | [] | 0.55/0.09/0.36 | 22 | 0/0 | 13/35 | 0 | 6 | 0 | stuck_repeating | plan:collect(W1) (0.59) |
| 20260927-124916 | tu93 | 0 | 0 | [] | [] | 0.55/0.32/0.14 | 37 | 0/2 | 0/36 | 0 | 6 | 0 | plans_ignored | DOWN (0.32) |
| 20260927-124916 | vc33 | 1 | 0.194 | [30] | [0.05] | 0.44/0.56/0.0 | 59 | 1/3 | 0/0 | 0 | 1 | 0 | stuck_repeating | click(9@45,1) (0.48) |
| 20260927-124916 | wa30 | 0 | 0 | [] | [] | 0.62/0.1/0.27 | 89 | 0/8 | 0/0 | 0 | 8 | 0 | hypotheses_refuted | LEFT (0.51) |