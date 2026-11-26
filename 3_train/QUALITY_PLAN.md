# 화질 개선 계획 (2026-11-26, 재촬영 없이 먼저)

목표: 영상 스캔 → 3DGS → VR 에서 Hyperscape 급으로 선명하게. 대상은 복도 1, 실험은 한 번에 한 가지만.
기준선: Brush 1920 / 300만 / SH2 / 30,000 (그대로 보존). 결과는 모두 3_train/README.md 표에 기록.
옵션 이름은 리서치(문서·소스)에서 본 것이며 **실행 전 --help 로 다시 확인**.

## 왜 Hyperscape 보다 못한가 (요약)
- Hyperscape: 헤드셋으로 표면 가까이 5~10분 촬영(정확한 VIO 자세) → 서버 학습 → **클라우드 GPU 렌더링 스트리밍** (헤드셋 예산·압축 없음)
- 우리: 어두운 곳 영상(블러·롤링셔터·자동 노출) + 흰 벽(SIFT 특징점 부족 → 자세 오차) + 절반 해상도 학습 + Unity 원본 논문 렌더러
- 3DGS 는 자세 오차에 아주 민감: 회전 ±1° 노이즈로 SSIM 0.76 → 0.35 (arXiv 2403.20309). **흐림의 1순위 후보는 자세**

## 0. 진단 (GPU 거의 안 씀, 반나절)
```
python tools\diagnose.py --video <영상>\1.MP4 --ffprobe F:\Sample\tools\ffmpeg_tool\ffprobe.exe
python tools\diagnose.py --frames F:\Sample\4차\1_프레임\1a --model F:\Sample\4차\2_정합데이터\1b\data
```
- HLG/PQ 경고 → SDR 변환부터 (색 물빠짐·저대비). 대비 < 80 이면 로그/HDR 의심
- "960px 로는 상위 절반인데 원본에서는 하위 25%" 비율이 크면 → 추출에 `--sharp-res 1920`, `--abs-floor-pct 10`
- 재투영 오차 > 1px, 사진당 3D 점이 적은 사진 다수 → 정합 개선(2단계)
- 벽 정면 사진 < 15%, 천장 < 5%, 한 높이만 → 재촬영 없이는 한계 (Difix·깊이 prior 로 일부 보완)
- 촬영 설정 확인: Osmo 의 전자식 흔들림 보정(RockSteady/HorizonSteady)·디워프가 켜져 있었으면 프레임마다 화각이 바뀌어 단일 카메라 가정이 깨짐

## 1. 자세가 원인인지 확인 (1구간, 1~2시간)
gsplat `examples/simple_trainer.py mcmc` 로 같은 데이터를 `--pose_opt` 켬/끔 두 번 학습 → 켠 쪽이 확 선명하면 자세가 원인.
(LichtFeld·Brush v0.3 에는 자세 최적화 옵션이 없음. gsplat 설치는 PIPELINE_UPGRADE.md ③′)

## 2. 정합 개선 (자세가 원인이면 가장 큰 효과)
config.ps1 (새 작업 폴더에서):
```
$FeatureArgs = @("--FeatureExtraction.type", "ALIKED_N16ROT")
$MatcherArgs = @("--FeatureMatching.type", "ALIKED_LIGHTGLUE", "--SequentialMatching.overlap", "20", "--SequentialMatching.loop_detection", "1")
```
- `feature_extractor -h` 에 FeatureExtraction.type 이 없으면(ONNX 없는 빌드) SIFT 유지 +
  `--SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1`, 매칭 `--FeatureMatching.guided_matching 1`
- `$Mapper = "incremental"` 과 `"global"` 을 한 구간에서 비교 (재투영 오차·등록률·천장 점 수: diagnose.py --model)

## 3. 학습 (한 가지씩)
1. LichtFeld `--ppisp --bg-modulation` (train_all.ps1 기본값. 자동 노출 흔들림 + 어두운 곳 투명화 대응)
2. 해상도: 3060 Ti `--max-width=2560` + `--tile-mode=2`, 5070 `--max-width=3840` + `--tile-mode=4` (`$TrainExtraArgs`)
   - 학습 해상도 = 표현 가능한 최대 디테일 (Mip-Splatting). 1920 은 VR 근접 시 4K 의 절반
3. 개수 400~600만 (`--max-cap`), 전략 `--strategy igs+` 와 `mcmc` 비교 (igs+ 는 세부가 선명하나 촬영이 나쁘면 흰 안개)
4. 그래도 흰 벽·천장이 안개 같으면: DN-Splatter(nerfstudio, WSL2) 단안 깊이·법선 prior
5. 블러·롤링셔터가 크면: SpectacularAI 3dgs-deblur (WSL2, 학습 약 5배 느림)
- `--enable-mip` / gsplat `antialiased` 는 끔 (aras-p 렌더러가 지원 안 함)
- "반투명 95%" 는 MCMC·Brush 의 opacity 정규화 때문에 원래 높게 나옴 → 숫자보다 **빨간/분홍 배경에서 비치는지**로 판단

## 4. VR 쪽 (AVT, 영향 확인만)
- aras-p 품질 프리셋 Very High/High 인지 (Low 이하는 PSNR 35 로 뭉갬), 구간 하나만 넣어 A/B
- 배경 메쉬(avt_backdrop) 끈 화면과 비교: 스플랫이 반투명하면 실제로는 메쉬 텍스처가 보이는 것
- aras-p v2.0.0 은 VR 경로가 깨졌을 수 있다고 작성자가 경고 → 버전 고정. 대안 A/B: ninjamode/Unity-VR-Gaussian-Splatting, wuyize25/gsplat-unity

## 재촬영 시 (위로도 부족하면)
노출·셔터(1/250+)·ISO·WB 수동 고정, 전자식 흔들림 보정 끔, 조명 보강, 높이 2~3단 왕복, 벽 정면·천장·바닥 패스, 천천히 + 루프, 흰 벽에 임시 마커
