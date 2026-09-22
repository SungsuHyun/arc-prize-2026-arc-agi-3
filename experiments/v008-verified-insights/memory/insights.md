# Agent Insight Memory
updated: 2026-09-22T07:48:50+00:00

## Evaluation Criteria (v4)
1. 외적 보상 (Extrinsic Reward): 승리 목표(레벨 완료)에 직접적으로 다가갔는가? (+1.0 ~ -1.0)
   - 레벨 완료, 새로운 화면 상태 도달, 목표 객체에 가까워짐 → 플러스
   - 아무 변화도 없는 행동 반복, 같은 상태 맴돌기, 액션 낭비 → 마이너스
2. 내적 보상 (Intrinsic Reward - 탐색 및 유연성): 새로운 전략을 시도하거나 변수에 잘 대처했는가? (0.0 ~ +0.5)
   - 이전에 시도하지 않은 낯선 행동(호기심)에 가산점 부여
+ (v2) Add a criterion for 'Convergence Penalty' in the Extrinsic Reward section: Deduct points if the agent continues to use low-efficiency actions despite having identified a high-efficiency action (e.g., ACTION6@c0) that directly addresses the goal.
+ (v3) The current 'Convergence Penalty' criterion in v2 is effective but needs to be explicitly quantified or emphasized as a primary driver for negative scoring when the agent ignores a discovered high-efficiency action. Suggest adding: 'If the agent executes a known high-efficiency action less than X% o
+ (v4) The current v3 criteria mention a 'Convergence Penalty' but lack a specific quantitative threshold for deduction (e.g., 'If the agent executes a known high-efficiency action less than X% of total actions after discovery'). It is recommended to explicitly define this threshold (e.g., < 50%) and assig

## General Insights
- High action diversity in early turns indicates effective exploration, but optimal goal completion requires converging on the most efficient action identified.
- High action diversity is beneficial for initial exploration, but optimal performance requires converging on the most efficient action once identified.
- While exploring diverse actions is crucial for discovering effective strategies, optimal performance requires immediately converging on the single most efficient action once identified.
- Once an agent identifies a high-efficiency action in an unknown environment, it must converge on that action rather than maintaining high diversity, as continued exploration with suboptimal actions delays goal completion.
- Optimal performance in unknown environments requires not only discovering the most efficient action but also rapidly converging on it, as excessive exploration after discovery leads to inefficiency.
- In unknown environments, initial exploration is valuable, but optimal performance requires immediate convergence on the most efficient action once identified.
- In unknown environments, once a high-efficiency action is identified through exploration, the agent must immediately converge on it to maximize goal completion speed rather than maintaining unnecessary diversity.
- Once a high-efficiency action is identified for a specific goal, the agent should immediately converge on it rather than maintaining high action diversity.
- Once a high-efficiency strategy is discovered in an unknown environment, immediate convergence on that strategy is critical for optimal performance, as continued exploration with suboptimal actions significantly delays goal completion.
- In unknown environments, once a high-efficiency action is discovered, the agent must rapidly converge on it rather than continuing to explore suboptimal actions, as excessive exploration after discovery significantly delays goal completion.
- In unknown environments, once a high-efficiency action is identified through exploration, the agent must immediately converge on it to maximize goal achievement and avoid inefficiency from excessive random exploration.
- Once a high-efficiency action is discovered in an unknown environment, immediate convergence on the most critical target is required to maximize goal completion speed rather than maintaining unnecessary diversity.

## Verified Action Facts (code-measured: effect count/tried)
### ls20
- ACTION1: grows c3 100/106, shrinks c11 99/106
- ACTION2: shrinks c11 82/87, grows c3 80/87
- ACTION3: grows c3 78/81, shrinks c11 77/81
- ACTION4: shrinks c11 116/120, grows c3 113/120
### lf52
- ACTION1: shrinks c0 26/52, grows c1 24/52
- ACTION2: shrinks c0 29/50, grows c1 26/50
- ACTION3: grows c1 27/50, shrinks c0 25/50
- ACTION4: shrinks c0 30/50, grows c1 23/50
- ACTION6@c0: grows c1 53/70, shrinks c0 47/70
- ACTION6@c1: shrinks c0 32/58, grows c1 25/58
- ACTION7: shrinks c0 30/62, grows c1 25/62
### sk48
- ACTION1: moves c6 (+0,-6) 49/59, moves c0 (+0,-6) 44/59
- ACTION2: moves c6 (+0,+6) 45/53, moves c0 (+0,+6) 42/53
- ACTION3: grows c4 50/87, removes c1 33/87
- ACTION4: shrinks c4 54/59, adds c1 39/59
- ACTION6@c1: no change 15/15
- ACTION6@c14: no change 8/8
- ACTION6@c2: no change 5/5
- ACTION6@c3: no change 14/14
- ACTION6@c4: no change 4/4
- ACTION6@c6: no change 13/13
- ACTION6@c8: no change 5/5
- ACTION6@c9: no change 9/9
### vc33
- ACTION6@c0: shrinks c7 43/43, grows c4 40/43
- ACTION6@c11: shrinks c7 22/22, grows c4 21/22
- ACTION6@c14: shrinks c7 18/18, grows c4 18/18
- ACTION6@c4: shrinks c7 49/49, grows c4 49/49
- ACTION6@c5: shrinks c7 69/69, grows c4 68/69
- ACTION6@c7: shrinks c7 90/90, grows c4 88/90
- ACTION6@c9: shrinks c0 68/73, grows c0 68/73

## Next Strategy
- ls20: ACTION4 target=none — Continuing to grow c3 and shrink c11 appears to be the primary mechanism for progressing towards level completion based
- lf52: ACTION6 target=c0 — Focusing exclusively on ACTION6 applied to c0 maximizes the rate of size reduction for the target object, directly accel
- sk48: ACTION3 target=none — The agent must exclusively execute ACTION3 to consistently grow the c4 objects and achieve level completion without wast
- vc33: ACTION6 target=c7 — Targeting c7 provides the maximum shrink rate (90 units) for the goal object, ensuring the fastest possible level comple

## Game Notes
### ls20
- ACTION3: In game ls20, ACTION3 consistently moves object c12 left by 5 units and object c9 left by 5 units.
- ACTION4: In game ls20, ACTION4 reliably grows object c3 while simultaneously shrinking object c11.
- ACTION1: In game ls20, ACTION1 consistently grows object c3 and shrinks object c11 with high efficiency.
- ACTION4: In game ls20, ACTION4 reliably grows object c3 while simultaneously shrinking object c11 with the highest efficiency among tested actions.
### lf52
- ACTION6: In game lf52, applying ACTION6 to the largest object c0 is the most efficient method to reduce its size towards zero.
- ACTION7: Applying ACTION7 in this game reduces both the target object c0 and the small objects c1 simultaneously.
- ACTION6: Applying ACTION6 to c0 is the most efficient way to shrink the target object, reducing its size by an average of 12 cells per successful application in this game.
- ACTION6: Applying ACTION6 to c0 is the most efficient way to reduce its size, shrinking it by approximately 20 cells per use while growing c1.
- ACTION7: ACTION7 reduces the size of small c1 objects but does not directly shrink the target c0 as effectively as ACTION6@c0.
- ACTION6: Applying ACTION6 specifically to c0 yields the highest rate of size reduction for the target object compared to other actions in this game.
- ACTION6: Applying ACTION6 to object c0 yields the highest reduction rate (38 units) compared to other actions, making it the optimal strategy for completing the level.
- ACTION6: Applying ACTION6 to the largest object (c0) yields the highest reduction rate per turn compared to other actions in this specific game state.
### sk48
- ACTION3: This action consistently grows the large c4 objects, which is the key to completing levels in this game.
- ACTION4: This action shrinks the target c4 objects and adds c1 cells, working against the level completion goal.
- ACTION7: This action frequently shrinks the large c4 objects or has no effect, directly opposing the goal of growth.
- ACTION3: This action consistently grows c4 objects in approximately half of its executions, which is the primary mechanism for level completion.
- ACTION3: ACTION3 consistently grows the large c4 objects, which is the necessary condition for level completion in this game.
- ACTION3: This action consistently grows c4 objects and removes c1 objects in this specific game environment.
- ACTION3: This action consistently grows the target c4 objects in this specific game instance.
- ACTION3: Executing ACTION3 consistently grows the target c4 objects and removes distractor c1 objects, which is the primary mechanism for level completion in this game.
### vc33
- ACTION6: In this game, ACTION6 consistently shrinks object c7 and grows object c4 regardless of the target clicked, indicating a global state change mechanism.
- ACTION6: Targeting c7 with ACTION6 provides the maximum shrink rate for the goal object, ensuring the fastest possible level completion.
- ACTION6: Targeting cell c7 with ACTION6 shrinks the goal object c7 at a rate of 52/52, which is the most efficient method for level completion.
- ACTION6: Targeting object c7 with ACTION6 achieves the maximum shrink rate of 100% (68/68) for the goal object in this game.
- ACTION6: Targeting object c7 with ACTION6 achieves a 100% success rate in shrinking the target and growing the auxiliary object, making it the most efficient path to level completion.

## Reward Log
| game | turn | reward | note |
|---|---|---|---|
| sk48 | 300 | -0.65 | code=+0.50 The agent's primary goal is to grow the large 'c4' objects to complete levels. The evalua |
| sk48 | 325 | -0.42 | code=+0.58 The agent's goal is to grow the large 'c4' objects (Extrinsic Reward). The measured data |
| sk48 | 350 | -0.42 | code=+0.54 The agent has identified ACTION3 as the high-efficiency action for growing c4 objects (th |
| sk48 | 375 | -0.42 | code=+0.38 The agent's goal is to grow the large c4 objects (size 816) to complete the level. The 'A |
| sk48 | 400 | -0.65 | code=+0.44 The agent has identified ACTION3 as the high-efficiency action for growing c4 objects (50 |
| vc33 | 25 | -0.40 | code=+0.60 The agent executed ACTION6 twenty-five times consecutively. While the action itself is hi |
| vc33 | 51 | -0.25 | code=+0.60 The agent executed ACTION6 24 times, which is the correct action type for shrinking objec |
| vc33 | 76 | -0.35 | code=+0.47 The agent executed ACTION6 twenty-five times. While the action successfully shrinks the t |
| vc33 | 102 | -0.40 | code=+0.60 The agent executed ACTION6 twenty-four times targeting various colors. While the action s |
| vc33 | 127 | -0.45 | code=+0.34 The agent executed ACTION6 twenty-five times. While the action successfully shrinks the t |
| vc33 | 153 | -0.15 | code=+0.45 The agent executed ACTION6 twenty-four times targeting various objects. While the action  |
| vc33 | 178 | -0.50 | code=+1.50 The agent executed ACTION6 twenty-five times. While the action successfully progresses th |
| vc33 | 203 | +0.45 | code=+0.60 The agent executed ACTION6 twenty-five times. Analysis of the 'Measured by code' data rev |
| vc33 | 228 | -0.25 | code=+0.53 The agent executed ACTION6 23 times. While the action itself is highly effective for the  |
| vc33 | 253 | -0.15 | code=+0.49 The agent executed ACTION6 twenty-five times. While the action itself is highly effective |
| vc33 | 278 | -0.25 | code=+0.51 The agent executed ACTION6 23 times. Analysis of the 'Measured' data reveals that targeti |
| vc33 | 303 | +0.35 | code=+0.50 The agent executed ACTION6 25 times, which is the most efficient action identified for sh |
| vc33 | 328 | -0.15 | code=+0.60 The agent executed ACTION6 23 times. While the action itself is highly effective (shrinki |
| vc33 | 353 | +0.55 | code=+0.60 The agent executed ACTION6 twenty-five times. Analysis of the measured effects shows that |
| vc33 | 378 | -0.15 | code=+0.41 The agent executed ACTION6 23 times. While the action itself is highly efficient for shri |

## Criteria History
- v1→v2: Add a criterion for 'Convergence Penalty' in the Extrinsic Reward section: Deduct points if the agent continues to use low-efficiency actions despite having identified a high-efficiency action (e.g.,
- v2→v3: The current 'Convergence Penalty' criterion in v2 is effective but needs to be explicitly quantified or emphasized as a primary driver for negative scoring when the agent ignores a discovered high-eff
- v3→v4: The current v3 criteria mention a 'Convergence Penalty' but lack a specific quantitative threshold for deduction (e.g., 'If the agent executes a known high-efficiency action less than X% of total acti
