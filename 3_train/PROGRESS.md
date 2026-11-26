# 진행 일지 (QUALITY_PLAN 실행, 2026-11-27 밤~)

목표: 지금 영상 데이터로 VR 에서 깨끗하고 빈틈없는 스플랫. 대상 복도 1. 실험 결과 표는 `3_train/README.md`.

## 지금 상태
- [x] 저장소 `pipeline-upgrade` 클론 → `F:\Sample\tools\3DGS_repo` (예전 `F:\Sample\tools\pipeline` 은 git 아님, 그대로 둠)
- [x] 0단계 도구 버전·옵션 확인
- [ ] 0단계 diagnose.py (원래 정합 1b, 복도 1 전체)
- [ ] 1단계 자세 확인 — **LichtFeld 없음 → 막힘** (아래 시행착오)
- [ ] 2단계 재정합 (sequential + loop, global / incremental) → 순간이동 수 비교
- [ ] 3단계 Brush 2560 → 빈 곳 채우기 (tools/fill)

## 도구 확인 (2026-11-27)
| 도구 | 버전 | train_all / register_all 옵션 대조 |
|---|---|---|
| COLMAP | 4.2.1 (bd1fcf6, CUDA) | ImageReader.single_camera·camera_model, SequentialMatching.overlap·loop_detection, view_graph_calibrator, global_mapper 모두 있음. FeatureExtraction.type(ALIKED) 옵션도 있음(GPU용 DLL 없을 수 있음) |
| Brush | 0.3.0 | --total-steps, --max-splats, --sh-degree, --max-resolution, --export-every, --export-path 맞음. --eval-split-every / --eval-every 로 PSNR 평가 가능. **자세 최적화 옵션 없음** |
| LichtFeld | **없음** | `F:\Sample\tools\lfs` 에 build_log.txt 만. 11-26 빌드 시도에서 Strawberry Perl, CUDA 12.8, VS2022 C++ 없음으로 중단 (VS 는 2026 Community 만 있음) |
| Python | 3.12.7, OpenCV 5.0.0, numpy 2.5.4 | |
- COLMAP 은 한글 경로(F:\Sample\4차\up)에서 정상 동작 확인 (3장 feature_extractor 시험)
- GitHub 인증 정보가 이 PC 에 저장돼 있지 않아 push 는 자고 있는 동안 못 함 → 로컬 커밋만 쌓고 나중에 push

## 시행착오
- LichtFeld `--pose-opt` 비교(1단계)는 설치(VS2022 C++ 워크로드, CUDA 12.8, Strawberry Perl)가 필요해 보류. 규칙상 설치는 먼저 물어봄.
  대안 gsplat `--pose_opt` 도 torch+gsplat 설치가 필요해 보류.
  → 대신 "자세가 원인인지"를 **재정합 전후 / 순간이동 사진 제외 전후 Brush PSNR** 로 간접 확인

## 오늘 밤 계획 (GPU 작업은 하나씩)
1. diagnose.py 로 원래 1b 의 순간이동 전체 목록 (CPU)
2. 같은 프레임(4차 1a 1100장 사본 → up\1a)으로 재정합: sequential(overlap 20, loop) + global → up\1b
3. 같은 매칭 DB 로 incremental → 비교용 폴더. 재투영 오차·등록률·순간이동 수 비교
4. Brush 기준선 설정(1920/300만/SH2/30,000, 평가 8장마다 1장)으로 원래 1b vs 새 정합 PSNR 비교
5. 나은 정합으로 2560 학습 (8GB 에서 메모리 부족이면 상한 낮춤) → tools/fill 바탕층·시작점 보강
