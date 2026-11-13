# 복도 1: 바탕층 합치기 + 보강된 시작점으로 Brush 학습
# 사용: PowerShell 에서  powershell -ExecutionPolicy Bypass -File F:\Sample\6\c\1c_fill\run_fill.ps1
$ErrorActionPreference = "Continue"
$fill  = "F:\Sample\6\c\1c_fill"
$data  = "F:\Sample\6\b\1b\data_fill"
$brush = "F:\Sample\brush\brush_app.exe"
$g     = "G:\내 드라이브\3DGS\3_학습결과"
Start-Transcript -Path "$fill\run_log.txt" -Append

# 1) 기존 결과에 바탕층 합치기 (바로 비교용, 1~2분)
python "$fill\merge_ply.py" "$fill\brush_base_with_backdrop.ply" "$g\4차_Brush_기본\1.ply" "$fill\backdrop_corridor1.ply"
python "$fill\merge_ply.py" "$fill\difix_with_backdrop.ply" "$g\4차_colab\out\corridor1\ply\corridor1_difix.ply" "$fill\backdrop_corridor1.ply"

# 2) 보강된 시작점으로 학습 (사진은 원본 폴더를 그대로 연결해서 씀, 복사 안 함)
if (-not (Test-Path "$data\sparse\0\images.bin")) { Copy-Item "F:\Sample\6\b\1b\data\sparse\0\images.bin" "$data\sparse\0\" }
if (-not (Test-Path "$data\images")) { New-Item -ItemType Junction -Path "$data\images" -Target "F:\Sample\6\b\1b\data\images" | Out-Null }
Write-Host "=== Brush 학습 시작 $(Get-Date)"
& $brush $data --max-splats 3000000 --sh-degree 2 --max-resolution 1920 --export-every 10000 --export-path $fill
Write-Host "=== Brush 학습 끝 $(Get-Date)"

# 3) 학습 결과에도 바탕층 합치기
python "$fill\merge_ply.py" "$fill\fill_with_backdrop.ply" "$fill\export_30000.ply" "$fill\backdrop_corridor1.ply"
Stop-Transcript
