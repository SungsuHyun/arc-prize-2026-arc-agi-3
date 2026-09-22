# Agent Insight Memory
updated: 2026-09-22T07:09:15+00:00

## Evaluation Criteria (v3)
1. 외적 보상 (Extrinsic Reward): 승리 목표(레벨 완료)에 직접적으로 다가갔는가? (+1.0 ~ -1.0)
   - 레벨 완료, 새로운 화면 상태 도달, 목표 객체에 가까워짐 → 플러스
   - 아무 변화도 없는 행동 반복, 같은 상태 맴돌기, 액션 낭비 → 마이너스
2. 내적 보상 (Intrinsic Reward - 탐색 및 유연성): 새로운 전략을 시도하거나 변수에 잘 대처했는가? (0.0 ~ +0.5)
   - 이전에 시도하지 않은 낯선 행동(호기심)에 가산점 부여
+ (v2) 평가 기준에 '반복적인 위치 변경 (oscillation)'과 '의도 없는 크기 조절'이 레벨 완료 조건을 충족하지 못함을 명시적으로 포함해야 합니다. 또한, 특정 시퀀스나 조건 없이 상태 변수만 조작할 경우 보상을 극도로 낮게 부여하는 규칙을 강화해야 합니다.
+ (v3) 현재 기준에 '반복적인 위치 변경 (oscillation)'과 '의도 없는 크기 조절'이 레벨 완료 조건을 충족하지 못함을 명시적으로 포함해야 합니다. 또한, 특정 시퀀스나 조건 없이 상태 변수만 조작할 경우 보상을 극도로 낮게 부여하는 규칙을 강화해야 합니다.

## General Insights
- In puzzle games where the goal is 'completion', arbitrary manipulation of state variables (like size or position) without a clear path to the goal condition leads to stagnation.

## Game Notes
### ls20
- In ls20, level completion requires a specific sequence or condition to remove objects; arbitrary resizing or position oscillation without reducing object count is ineffective.
- Repeatedly moving objects back and forth or resizing them without a clear trigger prevents progress toward the goal.
- In ls20, arbitrary resizing or position oscillation without a clear trigger sequence prevents level completion.
- In ls20, oscillating object positions or resizing them without a specific trigger sequence prevents level completion and wastes actions.
- Arbitrary manipulation of state variables (size/position) without reducing object count leads to stagnation and negative reward.
- The goal is to remove objects from the board; manipulating state variables (size/position) without reducing object count leads to stagnation.
- In ls20, removing objects requires a specific trigger sequence; arbitrary resizing or position oscillation without reducing object count leads to stagnation.
- In ls20, arbitrary resizing or position oscillation without reducing object count leads to stagnation and prevents level completion.
### lf52
- Arbitrary size or position manipulation without a clear sequence towards level completion results in severe stagnation and negative reward.
- The goal is to complete levels, not to manipulate state variables like color/size for their own sake.
- Arbitrary size reduction or position adjustment without a clear merging sequence leads to severe stagnation and negative reward.
- The goal is level completion; manipulating state variables like color/size for their own sake is meaningless.
- Arbitrary resizing of background blocks (c0) or small blocks (c1) without a clear merging sequence leads to severe stagnation and negative reward.
- The only effective action is using ACTION6 to click and eliminate the small 'c1' blocks to progress towards level completion.
- Arbitrary resizing of any block without a clear merging sequence leads to severe stagnation.
### sk48
- Arbitrary resizing and position changes without a clear sequence towards level completion are considered stagnation.
- Oscillation of object positions does not contribute to the goal condition in this puzzle game.
- Manipulating state variables (size/position) without contributing to the goal condition leads to negative reward.
### vc33
- Arbitrary resizing and position shifting without a clear sequence leads to stagnation and negative rewards.
- The goal is completion; manipulating state variables like size or position without progressing towards the level condition is futile.
- Clicking or manipulating state variables (size/position) without a clear sequence to remove objects leads to stagnation and negative rewards.
- The goal is completion; arbitrary manipulation of the grid without progressing towards clearing it is futile.

## Reward Log
| game | turn | reward | note |
|---|---|---|---|
| sk48 | 300 | -1.00 | 외적 보상: 에이전트의 최근 행동은 'oscillation' (반복적인 위치 변경) 과 'arbitrary resizing' (의도 없는 크기 조절) 을 포함하고 있습니다. 평가 |
| sk48 | 325 | -1.00 | 외적 보상: 현재 레벨이 완료되지 않았으며 (level 0/8), 에이전트의 행동은 주로 객체의 위치를 오가거나 (oscillation) 크기를 임의로 조절하는 등 상태 변수만 조 |
| sk48 | 350 | -1.00 | 외적 보상: 현재 레벨이 완료되지 않았으며 (level 0/8), 에이전트의 최근 행동은 주로 객체의 위치를 오가거나 (oscillation) 크기를 임의로 조절하는 것에 집중하고 |
| sk48 | 375 | -1.00 | 외적 보상: 현재 레벨이 완료되지 않았으며 (level 0/8), 에이전트의 행동은 객체의 위치 변경과 크기 조절에 집중하고 있습니다. 'Accumulated Insights'와 |
| sk48 | 400 | -0.80 | 외적 보상: 현재 레벨이 완료되지 않았으며 (level 0/8), 에이전트의 행동은 주로 객체의 위치 변경 (oscillation) 과 크기 조절에 집중하고 있습니다. 'Accum |
| vc33 | 25 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (level 0/7 -> 0/7) 에 전혀 기여하지 못했습니다. 최근 25 번의 액션은 모두 상태 변수 (크기, 위치) 만 조작하는 반복적인 행동 |
| vc33 | 51 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (level 0/7 -> 1/7) 에 전혀 기여하지 못했습니다. 최근 기록에 따르면 'resize'와 'move' 행동을 반복하며 상태 변수만 조 |
| vc33 | 76 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (level 0/7 -> 1/7) 에 전혀 기여하지 못했습니다. 목표는 '색상 객체 제거'이지만, 에이전트는 단순히 객체의 크기를 줄이고 위치를  |
| vc33 | 102 | -0.95 | 외적 보상: 에이전트는 레벨 완료 조건을 충족하지 못했으며, 보드 상태 변화가 없음. 반복적인 위치 변경 및 의도 없는 크기 조절이 관찰되어 목표 달성으로 이어지지 않음. 내적 보 |
| vc33 | 127 | -1.00 | 외적 보상: 에이전트는 레벨 완료(0/7)에 전혀 근접하지 않고 있으며, 단순히 블록의 크기와 위치를 임의로 조절하는 반복적인 행동만 수행했습니다. 이는 '상태 변수만 조작'하고  |
| vc33 | 153 | -1.00 | 외적 보상: 에이전트는 레벨 완료 목표에 직접적으로 다가가지 못했습니다. 'window of 24 actions'로 대량의 액션을 수행했으나 levels gained=0이며, 상태 |
| vc33 | 178 | -1.00 | 외적 보상: 에이전트는 레벨 완료 조건을 충족하지 못했으며, 보드 상태에 아무런 변화도 발생하지 않았습니다. 목표 객체를 제거하거나 레벨을 진행하는 대신, 기존 객체의 위치와 크기 |
| vc33 | 204 | -1.00 | 에이전트는 레벨 완료 (level 0/7) 를 달성하지 못했고, 새로운 화면 상태도 생성되지 않았습니다. 최근 기록에 따르면 에이전트가 무작위 크기 조절과 위치 이동만 반복하고 있 |
| vc33 | 229 | -1.00 | 에이전트는 현재 레벨 0/7 상태에서 목표 패턴을 완성하기 위한 명확한 시퀀스 없이, 단순히 블록의 크기와 위치를 반복적으로 조정하고 있습니다. 이는 'Accumulated Ins |
| vc33 | 255 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (level 0/7 -> 1/7) 에 도달하지 못했으며, 새로운 화면 상태도 생성되지 않았습니다. 목표 객체와 관련된 명확한 진행이 없으므로 -1 |
| vc33 | 280 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (level 0/7 -> 1/7) 를 위한 명확한 진행 없이, 단순히 객체의 위치를 좌우로 오가며 크기만 미세하게 조절했습니다. 이는 '반복적인  |
| vc33 | 306 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (level 0/7) 를 위해 모든 색상을 제거해야 하는 목표를 가지고 있음에도 불구하고, 24 번의 연속된 클릭 (Action 6) 을 실행하여 |
| vc33 | 331 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (level 0/7 -> 1/7) 에 전혀 기여하지 않았습니다. 최근 25 번의 모든 액션이 'ACTION6'으로, 이는 이전 실패 기록에 따르면 |
| vc33 | 357 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (level 0/7 -> 1/7) 를 위한 목표 객체를 제거하지 않고, 단순히 화면을 조작하거나 상태 변수만 변경했습니다. 이는 '반복적인 위치  |
| vc33 | 382 | -1.00 | 외적 보상: 에이전트는 레벨 완료 (clear grid) 를 위해 필요한 객체 제거가 아닌, 상태 변수 (크기/위치) 만 임의로 조작하는 행동을 반복했습니다. 이는 'Accumul |

## Criteria History
- v1→v2: 평가 기준에 '반복적인 위치 변경 (oscillation)'과 '의도 없는 크기 조절'이 레벨 완료 조건을 충족하지 못함을 명시적으로 포함해야 합니다. 또한, 특정 시퀀스나 조건 없이 상태 변수만 조작할 경우 보상을 극도로 낮게 부여하는 규칙을 강화해야 합니다.
- v2→v3: 현재 기준에 '반복적인 위치 변경 (oscillation)'과 '의도 없는 크기 조절'이 레벨 완료 조건을 충족하지 못함을 명시적으로 포함해야 합니다. 또한, 특정 시퀀스나 조건 없이 상태 변수만 조작할 경우 보상을 극도로 낮게 부여하는 규칙을 강화해야 합니다.
