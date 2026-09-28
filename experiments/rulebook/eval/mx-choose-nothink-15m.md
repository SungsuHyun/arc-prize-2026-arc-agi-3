# rulebook eval — mx-choose-nothink-15m

runs 1, game-runs 25, score mean 1.061, level1 rate 0.12, level2 rate 0.08, prediction ok/unknown/mismatch 0.37/0.44/0.13

Failure taxonomy (last level played): budget_time/actions_budget 9, stuck_repeating 6, no_win_hypothesis 4, plans_ignored 2, mismatch_heavy 2, predictor_blind 1, game_over_loop 1

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| lp85 | 1 | 16.67±0.0 | 3/3 | no_win_hypothesis |
| vc33 | 1 | 9.36±0.0 | 2/2 | no_win_hypothesis |
| r11l | 1 | 0.5±0.0 | 1/1 | no_win_hypothesis |
| ar25 | 1 | 0±0.0 | 0/0 | plans_ignored |
| bp35 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| cd82 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| cn04 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| dc22 | 1 | 0±0.0 | 0/0 | plans_ignored |
| ft09 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| g50t | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ka59 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| lf52 | 1 | 0±0.0 | 0/0 | predictor_blind |
| ls20 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| m0r0 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| re86 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| s5i5 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| sb26 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| sc25 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| sk48 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| sp80 | 1 | 0±0.0 | 0/0 | game_over_loop |
| su15 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| tn36 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| tr87 | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| tu93 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| wa30 | 1 | 0±0.0 | 0/0 | stuck_repeating |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260928-011845 | ar25 | 0 | 0 | [] | [] | 0.82/0.07/0.08 | 53 | 0/0 | 0/40 | 0 | 7 | 1 | plans_ignored | DOWN (0.32) |
| 20260928-011845 | bp35 | 0 | 0 | [] | [] | 0.16/0.37/0.33 | 76 | 1/0 | 0/45 | 0 | 9 | 2 | mismatch_heavy | RIGHT (0.38) |
| 20260928-011845 | cd82 | 0 | 0 | [] | [] | 0.0/0.94/0.06 | 227 | 0/4 | 210/210 | 0 | 1 | 0 | stuck_repeating | plan:collect(W2) (0.93) |
| 20260928-011845 | cn04 | 0 | 0 | [] | [] | 0.51/0.04/0.46 | 49 | 0/0 | 3/6 | 0 | 8 | 1 | mismatch_heavy | DOWN (0.33) |
| 20260928-011845 | dc22 | 0 | 0 | [] | [] | 0.95/0.03/0.03 | 70 | 0/2 | 0/1 | 0 | 2 | 0 | plans_ignored | UP (0.33) |
| 20260928-011845 | ft09 | 0 | 0 | [] | [] | 0.71/0.24/0.05 | 78 | 2/0 | 0/0 | 0 | 4 | 1 | budget_time/actions_budget | click(9@38,38) (0.22) |
| 20260928-011845 | g50t | 0 | 0 | [] | [] | 0.87/0.1/0.04 | 89 | 0/3 | 0/0 | 0 | 6 | 1 | budget_time/actions_budget | DOWN (0.31) |
| 20260928-011845 | ka59 | 0 | 0 | [] | [] | 0.16/0.57/0.1 | 42 | 0/0 | 2/3 | 0 | 5 | 0 | budget_time/actions_budget | RIGHT (0.31) |
| 20260928-011845 | lf52 | 0 | 0 | [] | [] | 0.05/0.83/0.02 | 126 | 0/10 | 0/112 | 0 | 3 | 1 | predictor_blind | DOWN (0.25) |
| 20260928-011845 | lp85 | 3 | 16.667 | [7, 27, 18] | [1.0, 1.0, 1.0] | 0.35/0.65/0.0 | 34 | 2/0 | 12/22 | 0 | 3 | 0 | no_win_hypothesis |  (0.0) |
| 20260928-011845 | ls20 | 0 | 0 | [] | [] | 0.35/0.08/0.07 | 70 | 0/0 | 4/38 | 0 | 6 | 0 | budget_time/actions_budget | DOWN (0.36) |
| 20260928-011845 | m0r0 | 0 | 0 | [] | [] | 0.79/0.05/0.16 | 14 | 0/0 | 1/1 | 0 | 5 | 0 | budget_time/actions_budget | DOWN (0.29) |
| 20260928-011845 | r11l | 1 | 0.498 | [7] | [1.0] | 0.42/0.52/0.04 | 67 | 2/1 | 0/0 | 0 | 5 | 1 | no_win_hypothesis |  (0.0) |
| 20260928-011845 | re86 | 0 | 0 | [] | [] | 0.74/0.05/0.21 | 31 | 0/1 | 2/10 | 0 | 8 | 0 | budget_time/actions_budget | RIGHT (0.39) |
| 20260928-011845 | s5i5 | 0 | 0 | [] | [] | 0.0/0.94/0.06 | 69 | 1/8 | 0/0 | 0 | 5 | 1 | stuck_repeating | click(11@37,23) (0.41) |
| 20260928-011845 | sb26 | 0 | 0 | [] | [] | 0.58/0.32/0.09 | 65 | 1/0 | 0/0 | 0 | 6 | 0 | budget_time/actions_budget | click(14@58,19) (0.26) |
| 20260928-011845 | sc25 | 0 | 0 | [] | [] | 0.07/0.52/0.08 | 85 | 0/10 | 10/152 | 0 | 8 | 1 | budget_time/actions_budget | LEFT (0.29) |
| 20260928-011845 | sk48 | 0 | 0 | [] | [] | 0.09/0.75/0.16 | 43 | 0/7 | 0/2 | 0 | 7 | 0 | stuck_repeating | ACTION7 (0.53) |
| 20260928-011845 | sp80 | 0 | 0 | [] | [] | 0.04/0.95/0.01 | 74 | 2/26 | 11/68 | 0 | 7 | 6 | game_over_loop | DOWN (0.36) |
| 20260928-011845 | su15 | 0 | 0 | [] | [] | 0.12/0.86/0.01 | 81 | 7/1 | 0/0 | 0 | 1 | 0 | stuck_repeating | ACTION7 (0.88) |
| 20260928-011845 | tn36 | 0 | 0 | [] | [] | 0.64/0.33/0.03 | 70 | 4/0 | 0/0 | 0 | 3 | 1 | budget_time/actions_budget | click(1@42,36) (0.07) |
| 20260928-011845 | tr87 | 0 | 0 | [] | [] | 0.13/0.3/0.57 | 47 | 0/0 | 0/0 | 0 | 8 | 0 | no_win_hypothesis | LEFT (0.38) |
| 20260928-011845 | tu93 | 0 | 0 | [] | [] | 0.33/0.47/0.2 | 37 | 0/4 | 0/35 | 0 | 8 | 0 | stuck_repeating | DOWN (0.49) |
| 20260928-011845 | vc33 | 2 | 9.355 | [6, 21] | [1.0, 0.73] | 0.04/0.95/0.01 | 77 | 4/5 | 0/0 | 0 | 3 | 0 | no_win_hypothesis | click(9@56,12) (0.58) |
| 20260928-011845 | wa30 | 0 | 0 | [] | [] | 0.33/0.15/0.48 | 103 | 0/7 | 1/13 | 0 | 8 | 0 | stuck_repeating | DOWN (0.43) |