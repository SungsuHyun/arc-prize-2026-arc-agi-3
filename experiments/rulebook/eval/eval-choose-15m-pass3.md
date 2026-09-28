# rulebook eval — eval-choose-15m-pass3

runs 1, game-runs 25, score mean 0.767, level1 rate 0.24, level2 rate 0.04, prediction ok/unknown/mismatch 0.38/0.42/0.13

Failure taxonomy (last level played): budget_time/actions_budget 8, no_win_hypothesis 4, stuck_repeating 3, plans_ignored 3, predictor_blind 3, mismatch_heavy 2, hypotheses_refuted 1, game_over_loop 1

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| vc33 | 1 | 5.94±0.0 | 2/2 | no_win_hypothesis |
| m0r0 | 1 | 3.94±0.0 | 1/1 | no_win_hypothesis |
| tn36 | 1 | 3.57±0.0 | 1/1 | budget_time/actions_budget |
| lp85 | 1 | 2.78±0.0 | 1/1 | stuck_repeating |
| r11l | 1 | 2.56±0.0 | 1/1 | no_win_hypothesis |
| ka59 | 1 | 0.4±0.0 | 1/1 | no_win_hypothesis |
| ar25 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| bp35 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| cd82 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |
| cn04 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| dc22 | 1 | 0±0.0 | 0/0 | plans_ignored |
| ft09 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| g50t | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| lf52 | 1 | 0±0.0 | 0/0 | plans_ignored |
| ls20 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| re86 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| s5i5 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sb26 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sc25 | 1 | 0±0.0 | 0/0 | plans_ignored |
| sk48 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sp80 | 1 | 0±0.0 | 0/0 | game_over_loop |
| su15 | 1 | 0±0.0 | 0/0 | stuck_repeating |
| tr87 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| tu93 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| wa30 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260928-064149 | ar25 | 0 | 0 | [] | [] | 0.79/0.03/0.18 | 29 | 0/0 | 1/14 | 0 | 9 | 1 | budget_time/actions_budget | UP (0.34) |
| 20260928-064149 | bp35 | 0 | 0 | [] | [] | 0.32/0.37/0.12 | 58 | 1/0 | 2/53 | 0 | 8 | 2 | stuck_repeating | RIGHT (0.47) |
| 20260928-064149 | cd82 | 0 | 0 | [] | [] | 0.04/0.95/0.01 | 99 | 0/3 | 0/0 | 0 | 1 | 0 | hypotheses_refuted | DOWN (0.38) |
| 20260928-064149 | cn04 | 0 | 0 | [] | [] | 0.83/0.05/0.12 | 35 | 1/0 | 1/12 | 0 | 7 | 0 | budget_time/actions_budget | DOWN (0.37) |
| 20260928-064149 | dc22 | 0 | 0 | [] | [] | 0.91/0.06/0.03 | 78 | 0/0 | 0/76 | 0 | 2 | 0 | plans_ignored | DOWN (0.33) |
| 20260928-064149 | ft09 | 0 | 0 | [] | [] | 0.63/0.33/0.04 | 75 | 2/1 | 0/0 | 0 | 3 | 1 | budget_time/actions_budget | click(9@38,38) (0.15) |
| 20260928-064149 | g50t | 0 | 0 | [] | [] | 0.89/0.08/0.03 | 57 | 0/0 | 2/112 | 0 | 4 | 0 | budget_time/actions_budget | DOWN (0.21) |
| 20260928-064149 | ka59 | 1 | 0.397 | [84] | [0.11] | 0.27/0.32/0.05 | 64 | 0/0 | 7/68 | 0 | 5 | 0 | no_win_hypothesis |  (0.0) |
| 20260928-064149 | lf52 | 0 | 0 | [] | [] | 0.04/0.42/0.06 | 41 | 0/8 | 0/30 | 0 | 3 | 0 | plans_ignored | DOWN (0.34) |
| 20260928-064149 | lp85 | 1 | 2.778 | [12] | [1.0] | 0.03/0.97/0.0 | 31 | 0/0 | 2/13 | 0 | 1 | 0 | stuck_repeating | click(8@26,14) (0.58) |
| 20260928-064149 | ls20 | 0 | 0 | [] | [] | 0.47/0.16/0.37 | 13 | 0/0 | 2/12 | 0 | 7 | 0 | mismatch_heavy | DOWN (0.31) |
| 20260928-064149 | m0r0 | 1 | 3.935 | [33] | [0.83] | 0.82/0.13/0.02 | 17 | 2/2 | 2/4 | 0 | 2 | 0 | no_win_hypothesis | UP (0.25) |
| 20260928-064149 | r11l | 1 | 2.561 | [30] | [0.54] | 0.13/0.63/0.23 | 30 | 0/2 | 0/0 | 0 | 8 | 0 | no_win_hypothesis |  (0.0) |
| 20260928-064149 | re86 | 0 | 0 | [] | [] | 0.71/0.05/0.22 | 54 | 0/1 | 5/73 | 0 | 8 | 0 | budget_time/actions_budget | UP (0.35) |
| 20260928-064149 | s5i5 | 0 | 0 | [] | [] | 0.03/0.91/0.06 | 67 | 3/4 | 0/0 | 0 | 5 | 1 | predictor_blind | click(11@37,23) (0.3) |
| 20260928-064149 | sb26 | 0 | 0 | [] | [] | 0.17/0.63/0.2 | 30 | 0/1 | 0/0 | 0 | 6 | 0 | predictor_blind | click(14@58,19) (0.2) |
| 20260928-064149 | sc25 | 0 | 0 | [] | [] | 0.29/0.29/0.08 | 75 | 2/9 | 0/100 | 0 | 7 | 1 | plans_ignored | DOWN (0.27) |
| 20260928-064149 | sk48 | 0 | 0 | [] | [] | 0.0/0.67/0.33 | 14 | 0/1 | 0/14 | 0 | 5 | 0 | predictor_blind | ACTION7 (0.36) |
| 20260928-064149 | sp80 | 0 | 0 | [] | [] | 0.06/0.93/0.01 | 56 | 1/4 | 17/48 | 0 | 7 | 5 | game_over_loop | plan:reach(W7) (0.21) |
| 20260928-064149 | su15 | 0 | 0 | [] | [] | 0.29/0.68/0.02 | 41 | 0/0 | 0/0 | 0 | 1 | 0 | stuck_repeating | ACTION7 (0.76) |
| 20260928-064149 | tn36 | 1 | 3.571 | [10] | [1.0] | 0.8/0.16/0.05 | 45 | 0/2 | 0/0 | 0 | 3 | 0 | budget_time/actions_budget | click(1@36,8) (0.06) |
| 20260928-064149 | tr87 | 0 | 0 | [] | [] | 0.5/0.39/0.11 | 18 | 0/0 | 5/17 | 0 | 2 | 0 | budget_time/actions_budget | DOWN (0.33) |
| 20260928-064149 | tu93 | 0 | 0 | [] | [] | 0.0/0.4/0.6 | 10 | 0/0 | 2/8 | 0 | 5 | 0 | mismatch_heavy | DOWN (0.2) |
| 20260928-064149 | vc33 | 2 | 5.937 | [34, 20] | [0.04, 0.81] | 0.31/0.69/0.0 | 54 | 1/2 | 0/0 | 0 | 2 | 0 | no_win_hypothesis |  (0.0) |
| 20260928-064149 | wa30 | 0 | 0 | [] | [] | 0.14/0.18/0.29 | 124 | 0/0 | 46/124 | 0 | 8 | 0 | budget_time/actions_budget | plan:collect(W1) (0.33) |