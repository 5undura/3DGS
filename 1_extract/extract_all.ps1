# 1단계: 영상 -> 프레임 추출
# 사용법 (PowerShell, 저장소 폴더에서):  .\1_extract\extract_all.ps1
# 결과: $Work\Na\f_<순번>_<원본프레임번호>.jpg
# 추출은 Na_tmp 에 하고 끝까지 성공해야 Na 로 이름을 바꿈 (중간에 끊긴 추출을 "이미 추출됨"으로 착각하지 않게)
. "$PSScriptRoot\..\config.ps1"
if (-not $ExtractArgs) { $ExtractArgs = @() }
if (-not $VideoDir) { $VideoDir = $Work }   # 영상이 다른 폴더에 있으면 config.ps1 의 $VideoDir 로 지정
New-Item -ItemType Directory -Force $Work | Out-Null

$log = Join-Path $Work "night_log.txt"
foreach ($n in $Sections) {
    $video = Join-Path $VideoDir "$n.MP4"
    $out   = Join-Path $Work "${n}a"
    $tmp   = Join-Path $Work "${n}a_tmp"
    if (-not (Test-Path -LiteralPath $video)) { Write-Host "[$n] 영상 없음: $video (건너뜀)"; continue }
    if ((Test-Path -LiteralPath $out) -and @(Get-ChildItem -LiteralPath $out -Filter *.jpg).Count -gt 0) {
        Write-Host "[$n] 이미 추출됨 (건너뜀)"; continue
    }
    if (Test-Path -LiteralPath $out) {
        # 빈 Na 폴더가 남아 있으면 이름 바꾸기가 막히므로 옆으로 치움 (삭제 안 함)
        Rename-Item -LiteralPath $out -NewName "${n}a_empty_$(Get-Date -Format yyyyMMdd_HHmmss)"
    }
    if (Test-Path -LiteralPath $tmp) {
        # 지난번에 끊긴 추출의 중간 결과 -> 옆으로 치우고 처음부터 (삭제 안 함)
        $old = "${n}a_failed_$(Get-Date -Format yyyyMMdd_HHmmss)"
        Rename-Item -LiteralPath $tmp -NewName $old
        Write-Host "[$n] 끊긴 추출 흔적을 $old 로 옮김"
    }
    & $Python "$PSScriptRoot\pick_best.py" $video $tmp @ExtractArgs
    if ($LASTEXITCODE -ne 0) { "[$n] 추출 실패 (종료 코드 $LASTEXITCODE, 중간 결과: $tmp)" | Tee-Object -FilePath $log -Append; continue }
    Rename-Item -LiteralPath $tmp -NewName "${n}a"
    $count = @(Get-ChildItem -LiteralPath $out -Filter *.jpg).Count
    "[$n] 추출 프레임: $count (옵션: $($ExtractArgs -join ' '))" | Tee-Object -FilePath $log -Append
}
