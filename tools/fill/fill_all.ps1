# 빈 곳 채우기를 여러 구간에 한 번에: 분석(바탕층 + 시작점 보강) -> Brush 학습 -> 바탕층 합치기
# 사용 (PowerShell):  powershell -ExecutionPolicy Bypass -File <저장소>\tools\fill\fill_all.ps1
# 끝난 단계는 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 함. 구간당 약 30~40분.
$ErrorActionPreference = "Continue"
$Sections = 2..6
$Work   = "F:\Sample\6"                     # b\Nb\data, c\Nc 가 있는 폴더
$Brush  = "F:\Sample\brush\brush_app.exe"
$Py     = "python"
$Here   = $PSScriptRoot

Start-Transcript -Path (Join-Path $Work "fill_all_log.txt") -Append
& $Py -c "import scipy, cv2, numpy" 2>$null
if ($LASTEXITCODE -ne 0) { & $Py -m pip install scipy opencv-python numpy matplotlib }

foreach ($n in $Sections) {
    $data = Join-Path $Work "b\${n}b\data"
    $fill = Join-Path $Work "b\${n}b\data_fill"
    $out  = Join-Path $Work "c\${n}c_fill"
    if (-not (Test-Path "$data\sparse\0\images.bin")) { Write-Host "[$n] $data\sparse\0 없음 (건너뜀)"; continue }
    Write-Host "===== [$n] 시작 $(Get-Date)"

    # 1) 분석: 바탕층 + 보강 시작점 (수 분)
    if (-not (Test-Path "$out\backdrop.ply")) {
        & $Py "$Here\fill_section.py" $data $fill $out 6
        if ($LASTEXITCODE -ne 0) { Write-Host "[$n] 분석 실패 (건너뜀)"; continue }
    }
    if (-not (Test-Path "$fill\images")) { New-Item -ItemType Junction -Path "$fill\images" -Target "$data\images" | Out-Null }

    # 2) Brush 학습 (기존 설정 그대로)
    if (-not (Test-Path "$out\export_30000.ply")) {
        Write-Host "[$n] Brush 학습 시작 $(Get-Date)"
        & $Brush $fill --max-splats 3000000 --sh-degree 2 --max-resolution 1920 --export-every 10000 --export-path $out
        Write-Host "[$n] Brush 학습 끝 $(Get-Date)"
    }

    # 3) 바탕층 합치기
    if ((Test-Path "$out\export_30000.ply") -and -not (Test-Path "$out\fill_with_backdrop.ply")) {
        & $Py "$Here\merge_ply.py" "$out\fill_with_backdrop.ply" "$out\export_30000.ply" "$out\backdrop.ply"
    }
    Write-Host "===== [$n] 끝 $(Get-Date)"
}
Stop-Transcript
