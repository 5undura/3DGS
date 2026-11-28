# 정합 빠르게: 1차 솎아 정합(pass1_stride) + 2차 global (2026-11-28)

합성 복도 v4 (261장), `run_e2e.py` 로 실제 register 단계 코드 그대로, COLMAP 대신 `colmap_shim.py` (pycolmap 4.2.1, CPU 4스레드).
데이터: `/tmp/claude-0/-home-user-3DGS/5698f89c-b779-591a-9900-84c73f83daf3/scratchpad/e2e_speed/` (지워질 수 있음).
"틀림" = 정답에 Sim3 로 맞춘 뒤 1 m 넘게 떨어진 사진 (`eval_e2e.py` sim3_bad1). 순간이동 = `diagnose.find_teleports`.

## 결론

1. **`pass1_stride: 3` 을 기본으로** (예전 1). 1차 147 s → 26 s, 전체 713 s → 482 s. 품질은 같거나 더 좋음:
   2차 incremental 을 seed 0·1·2 로 돌리면 stride 3·5 는 6번 모두 255장·틀림 0·순간이동 0, **stride 1 은 3번 중 2번 접힘** (71·97장 틀림, 순간이동 10).
   stride 5 도 같았지만 기준 자세가 157/261 (60%) 로 "절반 미만이면 검증 안 함" 문턱에 가까워 3 을 고름.
2. **`pass2_mapper: global` 은 기본으로 안 씀 (실험용, 위험)**. 2차 200~350 s → 55~65 s 로 빠르고 261장 모두 등록하지만,
   seed 9번 중 틀림 0 은 3번. 2~4장 틀림·순간이동 4~6 이 흔하고(diagnose 가 빼 줌), s3 seed 1 은 121장 틀림으로 크게 접힘.
   s3 seed 2 는 모델 안은 맞는데(틀림 3) 제약에 맞추기가 어긋나 도면 기준 243장이 1 m 넘게 틀림.
3. RESULTS.md 의 "두 번 정합 255/0/0" 은 **재현 안 됨**: 같은 코드를 다시 돌리면 매칭이 조금 달라져(짝 15165 → 15177) 2차 incremental 이 접힘.
   남은 가짜 짝(95개)이 원인으로 보임. stride 3·5 는 기준 자세가 없는 짝을 더 많이 지워(기준 없음 삭제 936 → 3574·6202) 남은 가짜 짝이 79·76.

## 1. 끝까지 (seed 0, 한 번씩)

| 실행 | 등록 | 틀림 >1 m | 순간이동 | 남은 가짜 짝 / 짝 | 1차 모델 | 기준 자세 (보간) | 특징 | seq | 1차 | spatial | 검증 | 2차 | undistort | 전체 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 incremental (예전 기본) | 255 | **71** | **10** | 95 / 8670 | [142, 133] | 248 (0) | 30 | 20 | 147 | 168 | 38 | 304 | 3 | 713 |
| s3 incremental | 255 | 0 | 0 | 79 / 7266 | [43, 41] /88 | 212 (139) | 28 | 18 | 26 | 148 | 30 | 226 | 3 | 482 |
| s5 incremental | 255 | 0 | 0 | 76 / 5945 | [22, 20] /53 | 157 (125) | 29 | 18 | 8 | 146 | 27 | 181 | 3 | 415 |
| s1 global | 261 | 0 | 0 | 95 | 같음 | 같음 | | | | | | 29+30 | 3 | 약 470* |
| s3 global | 261 | 2 (diagnose 뒤 0) | 4 | 79 | | | | | | | | 23+33 | 3 | 약 315* |
| s5 global | 261 | 3 (diagnose 뒤 0) | 4 | 76 | | | | | | | | 20+40 | 4 | 약 280* |

- 시간은 초. global 2차 = view_graph_calibrator + global_mapper. *global 전체 = 같은 실행의 incremental 전체에서 2차만 바꾼 값
  (global 실행은 incremental 실행 폴더를 복사해 `pass2_mapper` 만 바꿔 2차부터 다시 돈 것 — register 의 "2차 정합 방식이 바뀜" 경로 확인).
- 이전 RESULTS.md pv (같은 설정 s1, 다른 실행): 255장·틀림 0, 2차 221 s.
- global 은 6면 벽 모두 맞춤 (incremental 정상 실행은 끝 벽 포함 6면, s1 접힌 실행은 5면).

## 2. 2차만 seed 바꿔서 (같은 db_verified.db, `pass2_sweep.py`)

틀림 >1 m / 순간이동 (등록 incremental 255, global 261). seed 0 은 1절.

| 1차 | 2차 | seed 0 | seed 1 | seed 2 | 2차 시간 (s) |
|---|---|---|---|---|---|
| stride 1 | incremental | **71 / 10** | 0 / 0 | **97 / 10** | 218~350 |
| stride 3 | incremental | 0 / 0 | 0 / 0 | 0 / 0 | 201~226 |
| stride 5 | incremental | 0 / 0 | 0 / 0 | 0 / 0 | 158~181 |
| stride 1 | global | 0 / 0 | 0 / 0 | 4 / 6 | 57~59 |
| stride 3 | global | 2 / 4 | **121 / 5** | 3 / 5 (도면 기준 243) | 56~63 |
| stride 5 | global | 3 / 4 | 2 / 4 | 0 / 0 | 54~60 |

## 3. 실제 데이터 시간 (추정, 측정 아님)

방 하나 1,380장, 사용자 PC 한 번 정합 약 9 h. v4 의 예전 한 번 정합 mapper(576 s)를 9 h 로 놓고 단계 비율로만 환산 (매칭·GPU 차이 무시):

| 설정 | 1차 | 2차 | 정합 합 (추정) |
|---|---|---|---|
| stride 1 + incremental (예전) | 0.26 → 2.3 h | 0.38~0.53 → 3.5~4.8 h | 약 6~7 h |
| **stride 3 + incremental (새 기본)** | 0.045 → 0.4 h | 0.35~0.39 → 3.2~3.5 h | **약 3.6~3.9 h** |
| stride 3 + global (실험용) | 0.4 h | 0.10 → 0.9 h | 약 1.3 h |

- incremental 은 장수에 초선형 (여기서 사진 1/3 → 1차 1/5.7) 이라 1,380장에서는 1차 절감이 더 클 수 있음. spatial 매칭(여기 150 s, CPU)은 별도.

## 실행 방법

```
python run_e2e.py $V $W/s3 --venv $PY --pass1-stride 3                       # 약 8분
cp -a $W/s3 $W/s3g && mv $W/s3g/sections/v4/data $W/s3g/sections/v4/data_inc
python run_e2e.py $V $W/s3g --venv $PY --pass1-stride 3 --pass2-mapper global --until register
python run_e2e.py $V $W/s3g --venv $PY --pass1-stride 3 --pass2-mapper global --rerun diagnose
python eval_e2e.py $V $W/s3 $W/s3g
$PY/bin/python pass2_sweep.py $V $W/s3 incremental 1 $W/sweep/s3_inc_1   # 2차만 seed 바꿔서
```
