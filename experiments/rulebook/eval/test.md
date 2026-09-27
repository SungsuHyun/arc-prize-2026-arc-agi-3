# rulebook eval — test

runs 1, game-runs 6, score mean 2.077, level1 rate 0.67, level2 rate 0.17, prediction ok/unknown/mismatch 0.47/0.4/0.06

Failure taxonomy (last level played): budget_time/actions_budget 2, predictor_blind 2, no_win_hypothesis 1, budget_level_action_cap 1

| game | n | score mean±sd | levels mean/max | taxonomy |
|---|---|---|---|---|
| vc33 | 1 | 8.44±0.0 | 2/2 | predictor_blind |
| tn36 | 1 | 2.67±0.0 | 1/1 | budget_time/actions_budget |
| lp85 | 1 | 0.83±0.0 | 1/1 | predictor_blind |
| m0r0 | 1 | 0.52±0.0 | 1/1 | no_win_hypothesis |
| r11l | 1 | 0±0.0 | 0/0 | budget_time/actions_budget |
| ls20 | 1 | 0±0.0 | 0/0 | budget_level_action_cap |

| run | game | L | score | actions/level | eff | ok/unk/mis | decisions | fallback/veto | plans | submits | reviews | GO | taxonomy | top label (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260927-060249 | tn36 | 1 | 2.671 | [37] | [0.75] | 0.83/0.11/0.05 | 92 | 20/14 | 0/0 | 0 | 6 | 0 | budget_time/actions_budget | label 'click(1@36, (0.13) |
| 20260927-060249 | lp85 | 1 | 0.835 | [5] | [1.0] | 0.33/0.62/0.05 | 42 | 0/2 | 0/2 | 0 | 3 | 1 | predictor_blind | click(8@26,14) (1.0) |
| 20260927-060249 | r11l | 0 | 0 | [] | [] | 0.4/0.56/0.04 | 96 | 1/0 | 0/0 | 0 | 5 | 1 | budget_time/actions_budget | click(15@18,38) (0.23) |
| 20260927-060249 | vc33 | 2 | 8.438 | [28, 10] | [0.06, 1.0] | 0.26/0.74/0.0 | 46 | 2/1 | 0/0 | 0 | 2 | 0 | predictor_blind | click(9@56,12) (0.62) |
| 20260927-060249 | m0r0 | 1 | 0.518 | [91] | [0.11] | 0.77/0.06/0.13 | 65 | 2/0 | 1/11 | 0 | 10 | 0 | no_win_hypothesis | RIGHT (0.3) |
| 20260927-060249 | ls20 | 0 | 0 | [] | [] | 0.25/0.31/0.08 | 59 | 1/0 | 16/97 | 0 | 9 | 1 | budget_level_action_cap | DOWN (0.34) |