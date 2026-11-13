# 복도 1: 보강된 시작점으로 Brush 학습 + 바탕층 합치기
# 사용: PowerShell 에서  powershell -ExecutionPolicy Bypass -File F:\Sample\6\c\1c_fill\run_fill.ps1
$ErrorActionPreference = "Continue"
$fill  = "F:\Sample\6\c\1c_fill"
$data  = "F:\Sample\6\b\1b\data_fill"
$orig  = "F:\Sample\6\b\1b\data"
$brush = "F:\Sample\brush\brush_app.exe"
$g     = "G:\내 드라이브\3DGS\3_학습결과"
Start-Transcript -Path "$fill\run_log.txt" -Append

# 1) 학습 데이터 준비: 사진은 원본 폴더를 연결(복사 안 함), sparse 는 보강본 + 원본 images.bin
foreach ($fn in "cameras.bin","images.bin","frames.bin","rigs.bin") {
    if (-not (Test-Path "$data\sparse\0\$fn")) { Copy-Item "$orig\sparse\0\$fn" "$data\sparse\0\$fn" }
}
if (-not (Test-Path "$data\images")) { New-Item -ItemType Junction -Path "$data\images" -Target "$orig\images" | Out-Null }
Get-ChildItem "$data\sparse\0" | Format-Table Name, Length
if (-not (Test-Path "$data\sparse\0\images.bin")) { Write-Host "images.bin 없음 - 중단"; Stop-Transcript; exit 1 }

# 2) 보강된 시작점으로 학습 (약 1시간)
if (-not (Test-Path "$fill\export_30000.ply")) {
    Write-Host "=== Brush 학습 시작 $(Get-Date)"
    & $brush $data --max-splats 3000000 --sh-degree 2 --max-resolution 1920 --export-every 10000 --export-path $fill
    Write-Host "=== Brush 학습 끝 $(Get-Date)"
}

# 3) 바탕층 합치기 (이미 만든 건 건너뜀)
function Merge($out, $src) {
    if ((Test-Path $src) -and -not (Test-Path "$fill\$out")) { python "$fill\merge_ply.py" "$fill\$out" $src "$fill\backdrop_corridor1.ply" }
}
Merge "fill_with_backdrop.ply"       "$fill\export_30000.ply"
Merge "brush_base_with_backdrop.ply" "$g\4차_Brush_기본\1.ply"
Merge "difix_with_backdrop.ply"      "$g\4차_colab\out\corridor1\ply\corridor1_difix.ply"
Stop-Transcript
