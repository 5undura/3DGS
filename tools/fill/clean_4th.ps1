# 4차 구간 1~6: fill_with_backdrop.ply 에서 부유물 정리 -> fill_clean.ply (+ 지운 것만 fill_removed.ply)
# 사용: powershell -ExecutionPolicy Bypass -File F:\Sample\fill\clean_4th.ps1
$ErrorActionPreference = "Continue"
$Sections = 1..6
$Work = "F:\Sample\6"
$Here = "F:\Sample\fill"
Start-Transcript -Path (Join-Path $Work "clean_log.txt") -Append
foreach ($n in $Sections) {
    $data = Join-Path $Work "b\${n}b\data"
    $out  = Join-Path $Work "c\${n}c_fill"
    if (-not (Test-Path "$out\fill_with_backdrop.ply")) { Write-Host "[$n] fill_with_backdrop.ply 없음, 건너뜀"; continue }
    Write-Host "===== [$n] $(Get-Date)"
    python "$Here\clean_section.py" $data "$out\fill_with_backdrop.ply" "$out\fill_clean.ply" "$out\fill_removed.ply"
}
Stop-Transcript
