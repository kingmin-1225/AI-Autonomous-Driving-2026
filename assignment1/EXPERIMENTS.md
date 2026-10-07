# 과제 1 실험 기록

> 출처: Notion `Home / 3-2 / AI & Autonomous Driving` (2026-10-06 기준)

## 실험 결과

| Model | Epochs | Loss | Lap time (local / server) | Center (local / server) | 변경 사항 |
|---|---|---|---|---|---|
| baseline | 100 | — | 45.56s | 51.04 | — |
| ex1 | 100 (batch_size=256) | 1.717612 | 42.78s / 42.42s | 56.74 / 62.93 | model arch |
| ex2 | 100 (batch_size=256) | 0.897609 | 41.92s / DNF | 57.05 / — | base data(7) + 안전 주행(5) + 위험 주행(3) |
| ex3 | 50 | 0.074063 | 40.52s / 40.94s | 54.95 / 57.01 | 위험 주행(5) |
| ex4 | 50 | 0.345900 | 42.72s / 44.02s | 74.88 / 72.92 | 안전 주행(5) |
| ex5 | 50 | 0.513793 | 42.46s / 43.20s | 71.57 / 69.88 | 안전 주행(5) + 위험 주행(5) |

- 괄호 안 숫자는 사용한 주행 데이터(에피소드) 수.
- ex2는 서버 채점에서 완주 실패(DNF).

## 모델 구조 변경 (ex1~, `src/network.py`)

baseline:

```python
conv = Sequential(Conv2d(3, 8, 5, stride=4), ReLU(), AdaptiveAvgPool2d((4, 4)))
fc   = Sequential(Linear(8*4*4 + 7, n_classes))
```

변경 후:

```python
conv = Sequential(
    Conv2d(3, 16, 5, stride=2, padding=2),  BatchNorm2d(16), ReLU(),
    Conv2d(16, 32, 3, stride=2, padding=1), BatchNorm2d(32), ReLU(),
    Conv2d(32, 64, 3, stride=2, padding=1), BatchNorm2d(64), ReLU(),
    Conv2d(64, 64, 3, stride=1, padding=1), BatchNorm2d(64), ReLU(),
    AdaptiveAvgPool2d((4, 4)),
)
fc = Sequential(Linear(64*4*4 + 7, 128), ReLU(), Dropout(0.3), Linear(128, n_classes))
```

## 수집 데이터 주행 결과

### 안전 주행 (safe)

| # | 진행률 | 중앙선 유지 | 랩타임 |
|---|---|---|---|
| 1 | 100.00 | 71.77 | 42.50s |
| 2 | 100.00 | 75.17 | 42.78s |
| 3 | 100.00 | 74.65 | 42.50s |
| 4 | 100.00 | 75.67 | 42.68s |
| 5 | 100.00 | 76.54 | 42.58s |

### 위험 주행 (new risk)

| # | 진행률 | 중앙선 유지 | 랩타임 |
|---|---|---|---|
| 1 | 78.89 | 50.48 | 40.38s |
| 2 | 82.65 | 54.95 | 40.52s |
| 3 | 84.40 | 56.00 | 40.72s |
| 4 | 76.34 | 51.22 | 40.10s |
| 5 | 83.36 | 59.49 | 40.54s |

## ex2 loss 추이

<details>
<summary>펼치기 (100 epochs)</summary>

```
Epoch     1  loss: 19.861714
Epoch     2  loss: 11.447037
Epoch     3  loss: 8.350267
Epoch     4  loss: 6.332373
Epoch     5  loss: 5.395113
Epoch     6  loss: 4.313090
Epoch     7  loss: 3.726790
Epoch     8  loss: 3.142995
Epoch     9  loss: 2.912251
Epoch    10  loss: 2.589279
Epoch    11  loss: 2.813296
Epoch    12  loss: 2.475208
Epoch    13  loss: 2.013459
Epoch    14  loss: 1.639556
Epoch    15  loss: 1.604507
Epoch    16  loss: 1.484640
Epoch    17  loss: 1.637786
Epoch    18  loss: 2.129435
Epoch    19  loss: 1.792857
Epoch    20  loss: 2.123218
Epoch    21  loss: 1.879337
Epoch    22  loss: 2.083601
Epoch    23  loss: 1.806067
Epoch    24  loss: 1.755681
Epoch    25  loss: 1.472685
Epoch    26  loss: 1.124137
Epoch    27  loss: 1.450192
Epoch    28  loss: 1.566352
Epoch    29  loss: 1.578136
Epoch    30  loss: 2.215715
Epoch    31  loss: 1.773179
Epoch    32  loss: 1.677251
Epoch    33  loss: 1.641167
Epoch    34  loss: 1.415683
Epoch    35  loss: 1.477526
Epoch    36  loss: 1.233439
Epoch    37  loss: 1.475481
Epoch    38  loss: 1.515433
Epoch    39  loss: 1.687572
Epoch    40  loss: 1.356077
Epoch    41  loss: 1.615126
Epoch    42  loss: 1.479976
Epoch    43  loss: 1.612000
Epoch    44  loss: 1.389416
Epoch    45  loss: 1.322498
Epoch    46  loss: 1.439485
Epoch    47  loss: 1.648341
Epoch    48  loss: 1.743482
Epoch    49  loss: 1.482629
Epoch    50  loss: 0.907169
Epoch    51  loss: 1.243874
Epoch    52  loss: 1.106341
Epoch    53  loss: 1.544006
Epoch    54  loss: 1.386254
Epoch    55  loss: 1.507156
Epoch    56  loss: 1.317379
Epoch    57  loss: 1.119365
Epoch    58  loss: 1.056483
Epoch    59  loss: 1.418393
Epoch    60  loss: 1.357381
Epoch    61  loss: 1.105462
Epoch    62  loss: 1.351620
Epoch    63  loss: 1.298020
Epoch    64  loss: 0.954491
Epoch    65  loss: 1.070848
Epoch    66  loss: 1.204960
Epoch    67  loss: 1.449078
Epoch    68  loss: 1.099309
Epoch    69  loss: 1.295442
Epoch    70  loss: 1.206964
Epoch    71  loss: 1.257995
Epoch    72  loss: 0.907893
Epoch    73  loss: 1.415871
Epoch    74  loss: 1.322540
Epoch    75  loss: 1.306050
Epoch    76  loss: 0.917933
Epoch    77  loss: 0.890395
Epoch    78  loss: 1.299857
Epoch    79  loss: 1.152671
Epoch    80  loss: 1.483512
Epoch    81  loss: 1.207998
Epoch    82  loss: 1.185117
Epoch    83  loss: 0.787486
Epoch    84  loss: 1.333365
Epoch    85  loss: 1.488383
Epoch    86  loss: 1.550323
Epoch    87  loss: 1.083930
Epoch    88  loss: 1.013323
Epoch    89  loss: 1.048926
Epoch    90  loss: 0.990498
Epoch    91  loss: 1.044694
Epoch    92  loss: 1.076333
Epoch    93  loss: 0.919818
Epoch    94  loss: 0.902175
Epoch    95  loss: 1.486991
Epoch    96  loss: 0.861045
Epoch    97  loss: 0.928725
Epoch    98  loss: 0.865075
Epoch    99  loss: 0.674292
Epoch   100  loss: 0.897609
```

</details>
