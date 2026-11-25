# 한 번에: 프레임 추출 -> 정합 -> 학습 (config.ps1 설정대로)
# 사용: run_all.bat 더블클릭, 또는 PowerShell 에서  .\run_all.ps1
# 이미 끝난 단계·구간은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 진행됨.
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\config.ps1"
New-Item -ItemType Directory -Force $Work | Out-Null
$log = Join-Path $Work "night_log.txt"
"===== run_all 시작 $(Get-Date) / 구간 $($Sections -join ',') / 학습기 $Trainer / 프로필 $GpuProfile" | Tee-Object -FilePath $log -Append

& "$PSScriptRoot\1_extract\extract_all.ps1"
& "$PSScriptRoot\2_register\register_all.ps1"
& "$PSScriptRoot\3_train\train_all.ps1"

"===== run_all 끝 $(Get-Date)" | Tee-Object -FilePath $log -Append
