# 3차 촬영(F:\Sample\4\b1~b7) 빈 곳 채우기: 분석 -> Brush 학습 -> 바탕층 합치기
# 사용: powershell -ExecutionPolicy Bypass -File F:\Sample\fill\fill_3rd.ps1
$ErrorActionPreference = "Continue"
$Sections = 1..7
$Work   = "F:\Sample\4"
$Brush  = "F:\Sample\brush\brush_app.exe"
$Py     = "python"
$Here   = "F:\Sample\fill"

Start-Transcript -Path (Join-Path $Work "fill_3rd_log.txt") -Append
foreach ($n in $Sections) {
    $data = Join-Path $Work "b$n"
    $fill = Join-Path $Work "b${n}_fill"
    $out  = Join-Path $Work "c${n}_fill"
    if (-not (Test-Path "$data\sparse\0\images.bin")) { Write-Host "[$n] no sparse, skip"; continue }
    Write-Host "===== [$n] start $(Get-Date)"
    if (-not (Test-Path "$out\backdrop.ply")) {
        & $Py "$Here\fill_section.py" $data $fill $out 6
        if ($LASTEXITCODE -ne 0) { Write-Host "[$n] analysis failed, skip"; continue }
    }
    if (-not (Test-Path "$fill\images")) { New-Item -ItemType Junction -Path "$fill\images" -Target "$data\images" | Out-Null }
    if (-not (Test-Path "$out\export_30000.ply")) {
        Write-Host "[$n] brush start $(Get-Date)"
        & $Brush $fill --max-splats 3000000 --sh-degree 2 --max-resolution 1920 --export-every 10000 --export-path $out
        Write-Host "[$n] brush end $(Get-Date)"
    }
    if ((Test-Path "$out\export_30000.ply") -and -not (Test-Path "$out\fill_with_backdrop.ply")) {
        & $Py "$Here\merge_ply.py" "$out\fill_with_backdrop.ply" "$out\export_30000.ply" "$out\backdrop.ply"
    }
    Write-Host "===== [$n] end $(Get-Date)"
}
Stop-Transcript
