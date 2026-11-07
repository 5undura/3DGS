# Brush 학습 설정 실험 (해결 방향 3) - 여러 구간을 같은 설정으로 연속 학습
# 사용: 아래 $sections, $extra 를 고치고 PowerShell 에서 실행
$brush    = "F:\Sample\brush\brush_app.exe"
$work     = "F:\Sample\6"            # Nb\data 가 있는 작업 폴더
$outRoot  = "F:\Sample\exp3"
$name     = "E"                       # 결과 폴더 이름 앞부분 (E1, E2 ...)
$sections = 1..6

# 기본 설정 (기존 학습과 동일)
$base  = @("--max-splats","3000000","--sh-degree","2","--max-resolution","1920","--export-every","30000")
# 바꿀 설정 (E: 스케일 벌점 100배 + 성장 강화)
$extra = @("--scale-loss-weight","1e-6","--growth-grad-threshold","0.00002","--growth-select-fraction","0.2")

foreach ($n in $sections) {
    $out = Join-Path $outRoot "$name$n"
    New-Item -ItemType Directory -Force $out | Out-Null
    Write-Host "=== 구간 $n 시작 $(Get-Date)"
    & $brush (Join-Path $work "${n}b\data") @base @extra --export-path $out
    Write-Host "=== 구간 $n 끝 $(Get-Date)"
}
