# LichtFeld 학습 + 평가 + PLY 통계를 한 번에. 실험 비교용 (한 번에 한 가지만 바꿀 것)
# 사용: .\3_train\lfs_eval.ps1 -Data <Nb\data> -Out <결과 폴더> [-Cap 3000000] [-Width 1920] [-Extra @("--pose-opt")]
# 기본값 = train_all.ps1 의 LichtFeld 설정 (MCMC, PPISP, 배경 변조, SH2, 30,000, 8장마다 1장 평가)
#   단 상한은 Brush 기준선과 같은 300만 (train_all 의 3060ti 프로필은 150만)
# 옵션 이름은 v0.5.4.3 --help 로 확인 (2026-11-27)
# 결과: <Out>\splat_30000.ply, <Out>\lfs_log.txt, <Out>\metrics.txt(평가 요약), <Out>\stats.txt
param(
    [Parameter(Mandatory)][string]$Data,
    [Parameter(Mandatory)][string]$Out,
    [int]$Cap = 3000000,
    [int]$Width = 1920,
    [int]$Iter = 30000,
    [int]$Sh = 2,
    [string[]]$Extra = @()
)
. "$PSScriptRoot\..\config.ps1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force $Out | Out-Null
$log = Join-Path $Out "lfs_log.txt"
$ply = Get-ChildItem -LiteralPath $Out -Filter "splat_*.ply" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $ply) {
    $targs = @("-d", $Data, "-o", $Out, "--strategy=mcmc", "--max-cap=$Cap", "--iter=$Iter",
               "--sh-degree=$Sh", "--max-width=$Width", "--ppisp", "--bg-mode=modulation", "--export=ply",
               "--eval", "--test-every=8", "--headless") + $Extra
    "시작 $(Get-Date) / lichtfeld $($targs -join ' ')" | Out-File $log -Append -Encoding utf8
    $t = Get-Date
    & $LichtFeld @targs *>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    "끝 $(Get-Date) / 종료 코드 $code / $([int]((Get-Date) - $t).TotalMinutes)분" | Out-File $log -Append -Encoding utf8
    $ply = Get-ChildItem -LiteralPath $Out -Filter "splat_*.ply" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $ply) {
        $old = "$(Split-Path $Out -Leaf)_failed_$(Get-Date -Format yyyyMMdd_HHmmss)"
        Rename-Item -LiteralPath $Out -NewName $old
        Write-Host "실패 (종료 코드 $code) -> $old"; exit 1
    }
}
# 평가 요약: LichtFeld 가 남긴 metrics.csv 를 그대로 옮겨 적음
Get-ChildItem -LiteralPath $Out -Filter "metrics.csv" -Recurse | ForEach-Object { "== $($_.FullName)"; Get-Content -LiteralPath $_.FullName } |
    Tee-Object (Join-Path $Out "metrics.txt")
& $Python "$PSScriptRoot\..\tools\ply_stats.py" "run=$($ply.FullName)" | Tee-Object (Join-Path $Out "stats.txt")
