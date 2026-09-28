# rulebook eval — eval-choose-nothink-15m-pass4

runs 1, game-runs 25, score mean 0.94, level1 rate 0.16, level2 rate 0.08, prediction ok/unknown/mismatch 0.4/0.44/0.13

Failure taxonomy (last level played): budget_time/actions_budget 7, predictor_blind 7, stuck_repeating 4, plans_ignored 3, no_win_hypothesis 2, mismatch_heavy 1, game_over_loop 1

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| vc33 | 1 | 8.82±0.0 | 2/2 | predictor_blind |
| lp85 | 1 | 8.33±0.0 | 2/2 | no_win_hypothesis |
| tn36 | 1 | 3.57±0.0 | 1/1 | budget_time/actions_budget |
| ar25 | 1 | 2.78±0.0 | 1/1 | no_win_hypothesis |
| bp35 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| cd82 | 1 | 0±0.0 | 0/0 | predictor_blind |
| cn04 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| dc22 | 1 | 0±0.0 | 0/0 | plans_ignored |
| ft09 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| g50t | 1 | 0±0.0 | 0/0 | stuck_repeating |
| ka59 | 1 | 0±0.0 | 0/0 | plans_ignored |
| lf52 | 1 | 0±0.0 | 0/0 | predictor_blind |
| ls20 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| m0r0 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| r11l | 1 | 0±0.0 | 0/0 | predictor_blind |
| re86 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| s5i5 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sb26 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sc25 | 1 | 0±0.0 | 0/0 | plans_ignored |
| sk48 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| sp80 | 1 | 0±0.0 | 0/0 | game_over_loop |
| su15 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| tr87 | 1 | 0±0.0 | 0/0 | predictor_blind |
| tu93 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| wa30 | 1 | 0±0.0 | 0/0 | stuck_repeating |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260928-083444 | ar25 | 1 | 2.778 | [23] | [1.0] | 0.7/0.13/0.17 | 10 | 0/0 | 1/21 | 0 | 5 | 0 | no_win_hypothesis |  (0.0) |
| 20260928-083444 | bp35 | 0 | 0 | [] | [] | 0.18/0.53/0.21 | 34 | 3/0 | 1/33 | 0 | 7 | 0 | budget_time/actions_budget | ACTION7 (0.29) |
| 20260928-083444 | cd82 | 0 | 0 | [] | [] | 0.05/0.91/0.04 | 56 | 0/5 | 0/47 | 0 | 2 | 0 | predictor_blind | DOWN (0.38) |
| 20260928-083444 | cn04 | 0 | 0 | [] | [] | 0.82/0.06/0.12 | 11 | 0/0 | 1/7 | 0 | 4 | 0 | budget_time/actions_budget | DOWN (0.45) |
| 20260928-083444 | dc22 | 0 | 0 | [] | [] | 0.88/0.06/0.06 | 32 | 0/0 | 0/11 | 0 | 2 | 0 | plans_ignored | UP (0.34) |
| 20260928-083444 | ft09 | 0 | 0 | [] | [] | 0.68/0.29/0.03 | 31 | 1/2 | 0/0 | 0 | 1 | 0 | budget_time/actions_budget | click(2@10,12) (0.06) |
| 20260928-083444 | g50t | 0 | 0 | [] | [] | 0.82/0.07/0.05 | 33 | 0/0 | 0/32 | 0 | 2 | 0 | stuck_repeating | DOWN (0.42) |
| 20260928-083444 | ka59 | 0 | 0 | [] | [] | 0.23/0.42/0.13 | 26 | 0/0 | 0/17 | 0 | 4 | 0 | plans_ignored | DOWN (0.23) |
| 20260928-083444 | lf52 | 0 | 0 | [] | [] | 0.03/0.75/0.06 | 33 | 0/4 | 0/28 | 0 | 2 | 0 | predictor_blind | RIGHT (0.33) |
| 20260928-083444 | lp85 | 2 | 8.333 | [8, 38] | [1.0, 1.0] | 0.15/0.85/0.0 | 36 | 0/0 | 4/18 | 0 | 2 | 0 | no_win_hypothesis |  (0.0) |
| 20260928-083444 | ls20 | 0 | 0 | [] | [] | 0.78/0.05/0.17 | 20 | 0/0 | 5/31 | 0 | 7 | 0 | stuck_repeating | DOWN (0.45) |
| 20260928-083444 | m0r0 | 0 | 0 | [] | [] | 0.8/0.05/0.11 | 28 | 0/2 | 2/7 | 0 | 6 | 0 | budget_time/actions_budget | UP (0.32) |
| 20260928-083444 | r11l | 0 | 0 | [] | [] | 0.1/0.78/0.12 | 69 | 0/2 | 0/0 | 0 | 8 | 1 | predictor_blind | click(15@18,38) (0.09) |
| 20260928-083444 | re86 | 0 | 0 | [] | [] | 0.85/0.02/0.12 | 51 | 0/0 | 27/73 | 0 | 9 | 1 | budget_time/actions_budget | plan:collect(W2) (0.24) |
| 20260928-083444 | s5i5 | 0 | 0 | [] | [] | 0.02/0.9/0.08 | 59 | 1/11 | 0/0 | 0 | 6 | 1 | predictor_blind | click(11@37,23) (0.25) |
| 20260928-083444 | sb26 | 0 | 0 | [] | [] | 0.05/0.77/0.18 | 39 | 1/8 | 0/0 | 0 | 7 | 0 | predictor_blind | click(14@58,19) (0.18) |
| 20260928-083444 | sc25 | 0 | 0 | [] | [] | 0.19/0.29/0.28 | 56 | 0/6 | 0/100 | 0 | 9 | 1 | plans_ignored | DOWN (0.27) |
| 20260928-083444 | sk48 | 0 | 0 | [] | [] | 0.48/0.01/0.51 | 72 | 0/0 | 25/83 | 0 | 8 | 0 | mismatch_heavy | plan:collect(W2) (0.31) |
| 20260928-083444 | sp80 | 0 | 0 | [] | [] | 0.16/0.83/0.01 | 69 | 2/0 | 14/73 | 0 | 11 | 9 | game_over_loop | plan:collect(W1) (0.2) |
| 20260928-083444 | su15 | 0 | 0 | [] | [] | 0.08/0.91/0.01 | 110 | 0/0 | 0/0 | 0 | 1 | 0 | stuck_repeating | ACTION7 (0.91) |
| 20260928-083444 | tn36 | 1 | 3.571 | [17] | [1.0] | 0.58/0.32/0.09 | 78 | 6/13 | 0/0 | 1 | 8 | 1 | budget_time/actions_budget | click(9@54,7) (0.1) |
| 20260928-083444 | tr87 | 0 | 0 | [] | [] | 0.1/0.66/0.24 | 50 | 0/0 | 0/0 | 0 | 8 | 0 | predictor_blind | DOWN (0.36) |
| 20260928-083444 | tu93 | 0 | 0 | [] | [] | 0.68/0.17/0.15 | 35 | 0/0 | 6/68 | 0 | 9 | 1 | budget_time/actions_budget | RIGHT (0.26) |
| 20260928-083444 | vc33 | 2 | 8.82 | [17, 10] | [0.17, 1.0] | 0.06/0.94/0.0 | 67 | 1/1 | 0/0 | 0 | 2 | 0 | predictor_blind | click(9@56,12) (0.28) |
| 20260928-083444 | wa30 | 0 | 0 | [] | [] | 0.57/0.12/0.28 | 74 | 0/0 | 0/107 | 0 | 7 | 0 | stuck_repeating | RIGHT (0.45) |