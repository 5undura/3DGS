# 1단계: 영상 -> 프레임 추출
# 사용법 (PowerShell, 저장소 폴더에서):  .\1_extract\extract_all.ps1
# 결과: $Work\Na\f_<순번>_<원본프레임번호>.jpg
. "$PSScriptRoot\..\config.ps1"

$log = Join-Path $Work "night_log.txt"
foreach ($n in $Sections) {
    $video = Join-Path $Work "$n.MP4"
    $out   = Join-Path $Work "${n}a"
    if (-not (Test-Path $video)) { Write-Host "[$n] 영상 없음: $video (건너뜀)"; continue }
    if ((Test-Path $out) -and (Get-ChildItem $out -Filter *.jpg).Count -gt 0) {
        Write-Host "[$n] 이미 추출됨 (건너뜀)"; continue
    }
    & $Python "$PSScriptRoot\pick_best.py" $video $out
    if ($LASTEXITCODE -ne 0) { Write-Host "[$n] 추출 실패"; continue }
    $count = (Get-ChildItem $out -Filter *.jpg).Count
    "[$n] 추출 프레임: $count" | Tee-Object -FilePath $log -Append
}
