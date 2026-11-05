# 3단계: Brush 학습

Brush 앱(`brush_app.exe`)으로 학습함. CLI도 있음 (`brush_app.exe --help`).

## 순서
1. Brush 실행 → 데이터 폴더로 `Nb\data` 선택 (안에 `images`, `sparse\0` 이 있어야 함)
2. 아래 설정으로 학습
3. 내보낸 PLY는 `Nc` 폴더에 저장 (`export_30000.ply` 가 최종)

## 우리가 쓰는 설정
| 항목 | 값 | 비고 |
|---|---|---|
| 해상도 (최대 이미지 크기) | 1920 | 4K는 학습이 무너진 적 있음 (방 5: 스플랫 3.5만 개로 붕괴) |
| 최대 스플랫 수 | 3,000,000 | 복도는 6,000,000 버전도 있으나 VR에서 끊김 → 3M 권장 |
| SH 차수 | 2 | |
| 스텝 | 30,000 | 5,000 스텝마다 중간 PLY 저장 |

## 확인
- SuperSplat(https://superspl.at/editor)에 PLY를 끌어다 놓으면 바로 확인 가능
- Unity: aras-p UnityGaussianSplatting 플러그인으로 PLY → 에셋 변환
