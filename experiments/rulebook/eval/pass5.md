# rulebook eval — pass5

runs 1, game-runs 25, score mean 0.479, level1 rate 0.12, level2 rate 0.04, prediction ok/unknown/mismatch 0.38/0.37/0.16

Failure taxonomy (last level played): budget_time/actions_budget 7, no_win_hypothesis 5, predictor_blind 5, hypotheses_refuted 4, plan_unusable 2, mismatch_heavy 2

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| vc33 | 1 | 8.79±0.0 | 2/2 | no_win_hypothesis |
| lp85 | 1 | 2.78±0.0 | 1/1 | predictor_blind |
| m0r0 | 1 | 0.42±0.0 | 1/1 | budget_time/actions_budget |
| ar25 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |
| bp35 | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| cd82 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |
| cn04 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| dc22 | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| ft09 | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| g50t | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ka59 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |
| lf52 | 1 | 0±0.0 | 0/0 | plan_unusable |
| ls20 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| r11l | 1 | 0±0.0 | 0/0 | predictor_blind |
| re86 | 1 | 0±0.0 | 0/0 | plan_unusable |
| s5i5 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sb26 | 1 | 0±0.0 | 0/0 | predictor_blind |
| sc25 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| sk48 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| sp80 | 1 | 0±0.0 | 0/0 | predictor_blind |
| su15 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| tn36 | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| tr87 | 1 | 0±0.0 | 0/0 | no_win_hypothesis |
| tu93 | 1 | 0±0.0 | 0/0 | mismatch_heavy |
| wa30 | 1 | 0±0.0 | 0/0 | hypotheses_refuted |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260928-103213 | ar25 | 0 | 0 | [] | [] | 0.78/0.06/0.17 | 11 | 0/0 | 0/0 | 0 | 6 | 0 | hypotheses_refuted | LEFT (0.27) |
| 20260928-103213 | bp35 | 0 | 0 | [] | [] | 0.35/0.15/0.27 | 15 | 0/0 | 0/0 | 0 | 8 | 1 | no_win_hypothesis | LEFT (0.53) |
| 20260928-103213 | cd82 | 0 | 0 | [] | [] | 0.05/0.87/0.08 | 24 | 0/0 | 0/0 | 0 | 3 | 0 | hypotheses_refuted | RIGHT (0.25) |
| 20260928-103213 | cn04 | 0 | 0 | [] | [] | 0.72/0.06/0.22 | 13 | 0/0 | 2/2 | 0 | 6 | 0 | budget_time/actions_budget | DOWN (0.23) |
| 20260928-103213 | dc22 | 0 | 0 | [] | [] | 0.97/0.02/0.02 | 68 | 0/0 | 0/0 | 0 | 2 | 0 | no_win_hypothesis | DOWN (0.32) |
| 20260928-103213 | ft09 | 0 | 0 | [] | [] | 0.85/0.11/0.03 | 27 | 3/0 | 0/0 | 0 | 2 | 1 | no_win_hypothesis | click(8@38,38) (0.22) |
| 20260928-103213 | g50t | 0 | 0 | [] | [] | 0.91/0.02/0.05 | 68 | 1/0 | 1/1 | 0 | 8 | 1 | budget_time/actions_budget | DOWN (0.22) |
| 20260928-103213 | ka59 | 0 | 0 | [] | [] | 0.05/0.11/0.47 | 17 | 0/2 | 0/0 | 0 | 8 | 0 | hypotheses_refuted | UP (0.41) |
| 20260928-103213 | lf52 | 0 | 0 | [] | [] | 0.07/0.53/0.05 | 40 | 0/3 | 0/0 | 0 | 2 | 0 | plan_unusable | DOWN (0.38) |
| 20260928-103213 | lp85 | 1 | 2.778 | [7] | [1.0] | 0.01/0.99/0.0 | 44 | 1/5 | 3/10 | 0 | 2 | 1 | predictor_blind | click(8@26,14) (0.34) |
| 20260928-103213 | ls20 | 0 | 0 | [] | [] | 0.38/0.04/0.05 | 33 | 0/0 | 1/2 | 0 | 5 | 0 | budget_time/actions_budget | DOWN (0.21) |
| 20260928-103213 | m0r0 | 1 | 0.42 | [101] | [0.09] | 0.86/0.03/0.08 | 43 | 1/0 | 0/0 | 0 | 9 | 0 | budget_time/actions_budget | SPACE (1.0) |
| 20260928-103213 | r11l | 0 | 0 | [] | [] | 0.0/0.92/0.08 | 9 | 0/0 | 0/0 | 0 | 1 | 0 | predictor_blind | click(6@48,18) (0.44) |
| 20260928-103213 | re86 | 0 | 0 | [] | [] | 0.96/0.02/0.02 | 9 | 0/0 | 0/0 | 0 | 1 | 0 | plan_unusable | goto(4@9,24) (0.33) |
| 20260928-103213 | s5i5 | 0 | 0 | [] | [] | 0.09/0.73/0.18 | 9 | 0/0 | 0/0 | 0 | 2 | 0 | predictor_blind | click(11@43,23) (0.33) |
| 20260928-103213 | sb26 | 0 | 0 | [] | [] | 0.1/0.8/0.1 | 14 | 1/0 | 0/0 | 0 | 2 | 0 | predictor_blind | click(14@58,19) (0.29) |
| 20260928-103213 | sc25 | 0 | 0 | [] | [] | 0.11/0.39/0.25 | 18 | 0/0 | 4/6 | 0 | 7 | 0 | budget_time/actions_budget | RIGHT (0.33) |
| 20260928-103213 | sk48 | 0 | 0 | [] | [] | 0.3/0.15/0.55 | 15 | 0/1 | 2/7 | 0 | 8 | 0 | mismatch_heavy | UP (0.33) |
| 20260928-103213 | sp80 | 0 | 0 | [] | [] | 0.32/0.64/0.04 | 20 | 2/0 | 7/13 | 0 | 5 | 2 | predictor_blind | RIGHT (0.2) |
| 20260928-103213 | su15 | 0 | 0 | [] | [] | 0.6/0.4/0.0 | 15 | 1/1 | 0/0 | 0 | 0 | 0 | budget_time/actions_budget | ACTION7 (0.33) |
| 20260928-103213 | tn36 | 0 | 0 | [] | [] | 0.51/0.37/0.12 | 28 | 2/0 | 0/0 | 0 | 5 | 0 | budget_time/actions_budget | click(1@42,31) (0.11) |
| 20260928-103213 | tr87 | 0 | 0 | [] | [] | 0.35/0.35/0.3 | 17 | 0/0 | 0/0 | 0 | 6 | 0 | no_win_hypothesis | UP (0.53) |
| 20260928-103213 | tu93 | 0 | 0 | [] | [] | 0.0/0.43/0.57 | 5 | 0/0 | 1/2 | 0 | 3 | 0 | mismatch_heavy | DOWN (0.6) |
| 20260928-103213 | vc33 | 2 | 8.786 | [15, 17] | [0.22, 1.0] | 0.19/0.81/0.0 | 27 | 2/6 | 0/0 | 0 | 2 | 0 | no_win_hypothesis |  (0.0) |
| 20260928-103213 | wa30 | 0 | 0 | [] | [] | 0.08/0.17/0.33 | 11 | 0/0 | 0/0 | 0 | 4 | 0 | hypotheses_refuted | DOWN (0.55) |