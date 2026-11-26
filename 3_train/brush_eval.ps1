# Brush 학습 + 평가(PSNR/SSIM) + PLY 통계를 한 번에. 실험 비교용 (한 번에 한 가지만 바꿀 것)
# 사용: .\3_train\brush_eval.ps1 -Data <Nb\data> -Out <결과 폴더> [-Res 1920] [-Cap 3000000] [-Extra @("--opt","값")]
# 기본값 = 기준선 (1920 / 300만 / SH2 / 30,000). 8장마다 1장은 학습에서 빼고 평가 (--eval-split-every 8)
# 결과: <Out>\export_30000.ply, <Out>\eval_30000\*.png, <Out>\eval.txt, eval.csv, stats.txt, brush_log.txt
param(
    [Parameter(Mandatory)][string]$Data,
    [Parameter(Mandatory)][string]$Out,
    [int]$Res = 1920,
    [int]$Cap = 3000000,
    [int]$Steps = 30000,
    [int]$Sh = 2,
    [string[]]$Extra = @()
)
. "$PSScriptRoot\..\config.ps1"
$env:PYTHONIOENCODING = "utf-8"
$ply = Join-Path $Out "export_$Steps.ply"
New-Item -ItemType Directory -Force $Out | Out-Null
$log = Join-Path $Out "brush_log.txt"
if (-not (Test-Path -LiteralPath $ply)) {
    $targs = @($Data, "--max-splats", "$Cap", "--sh-degree", "$Sh", "--max-resolution", "$Res",
               "--total-steps", "$Steps", "--export-every", "$Steps", "--export-path", $Out,
               "--eval-split-every", "8", "--eval-every", "$Steps", "--eval-save-to-disk") + $Extra
    "시작 $(Get-Date) / brush $($targs -join ' ')" | Out-File $log -Append -Encoding utf8
    $t = Get-Date
    & $Brush @targs *>&1 | ForEach-Object { "$_" } | Out-File $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    "끝 $(Get-Date) / 종료 코드 $code / $([int]((Get-Date) - $t).TotalMinutes)분" | Out-File $log -Append -Encoding utf8
    if (-not (Test-Path -LiteralPath $ply)) {
        $old = "$(Split-Path $Out -Leaf)_failed_$(Get-Date -Format yyyyMMdd_HHmmss)"
        Rename-Item -LiteralPath $Out -NewName $old
        Write-Host "실패 (종료 코드 $code) -> $old"; exit 1
    }
}
& $Python "$PSScriptRoot\..\tools\eval_renders.py" (Join-Path $Out "eval_$Steps") $Data --csv (Join-Path $Out "eval.csv") | Tee-Object (Join-Path $Out "eval.txt")
& $Python "$PSScriptRoot\..\tools\ply_stats.py" "run=$ply" | Tee-Object (Join-Path $Out "stats.txt")
