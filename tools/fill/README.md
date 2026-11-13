# 빈 곳 채우기 (바닥·천장·벽 바탕층 + 시작점 보강)

## 바로 쓰기 (PC)
```powershell
powershell -ExecutionPolicy Bypass -File <저장소>\tools\fill\fill_all.ps1
```
`fill_all.ps1` 안의 `$Sections`(기본 2..6), `$Work`(기본 F:\Sample\6), `$Brush` 경로를 확인. 구간마다
`fill_section.py`(분석, 수 분) → Brush 학습(약 30분) → `merge_ply.py`(바탕층 합치기) 순서. 결과: `c\Nc_fill\fill_with_backdrop.ply`.
끝난 단계는 건너뛰므로 끊겨도 다시 실행하면 이어서 함. 진행 기록: `F:\Sample\6\fill_all_log.txt`, 구간별 진단 그림: `Nc_fill\fill_diag.png`.

## 개발 과정 스크립트 (복도 1 에서 단계별로 만든 것, fill_section.py 로 합쳐짐)

복도처럼 "상자" 모양인 구간에서, 스플랫이 옅거나 빈 곳(특히 천장)을 채우는 도구.

1. `load.py` – COLMAP `sparse/0` 읽기 → `scene.pkl`
2. `analyze2.py` – 바닥 평면(RANSAC)과 복도 방향 찾기 → `frame2.npz`, 진단 그림
3. `geom.py` – 바닥 점으로 복도 평면도(footprint) 만들기 → `foot.npz`
4. `backdrop.py` – 바닥·천장(높이 H)·벽(평면도 경계)에 샘플을 깔고, 사진에 투영해 색을 정함 (가까운 정면 사진 6장의 중간값)
5. `make_outputs.py` – ① 바탕층 스플랫 PLY (불투명·납작, 면 1.2cm 뒤) ② 원래 points3D + 샘플 점을 합친 보강 시작점
6. `merge_ply.py` – 학습 결과 PLY 와 바탕층 PLY 합치기
7. `run_fill.ps1` – PC 에서 합치기 + 보강 시작점으로 Brush 학습

복도 1 기준: 바닥 평면 잔차 중간값 0.006, 천장 높이 1.60 (COLMAP 단위, 카메라 높이 0.67), 바탕층 20만 개, 시작점 64만 → 85만 개.
COLMAP 분석 결과 바닥은 점이 충분하지만 **천장은 점이 거의 없음** → 천장이 비는 주원인.
