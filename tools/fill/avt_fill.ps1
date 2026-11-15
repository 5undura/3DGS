# 우리 4차 보강 학습 결과를 팀원 Unity 프로젝트(AVT)에 넣기
#   1) 구간마다 바탕층 없는 export_30000.ply 를 부유물 정리 -> nobd_clean.ply
#   2) 1.ply~6.ply 이름으로 모음 (하드링크라 용량 추가 없음)
#   3) 팀원의 place.py 로 Unity 좌표에 배치 + 공간별로 잘라 합침 -> merged_v4_fill.ply
#   4) AVT\SplatSource 에 복사 -> Unity 에서 Tools > Student Hall > 6 실행
# 사용: powershell -ExecutionPolicy Bypass -File F:\Sample\tools\fill\avt_fill.ps1
$ErrorActionPreference = "Continue"
$W    = "F:\Sample\4차"
$Here = "F:\Sample\tools\fill"
$AVT  = "E:\Unity\Projects\AVT"
$src  = "$W\3_학습결과\avt_src"
$out  = "$W\3_학습결과\avt_out"
Start-Transcript -Path "$W\기록\avt_fill_log.txt" -Append
New-Item -ItemType Directory -Force $src, $out, "$AVT\SplatSource" | Out-Null

foreach ($n in 1..6) {
    $data = "$W\2_정합데이터\${n}b\data"
    $c    = "$W\3_학습결과\${n}c_fill"
    if (-not (Test-Path "$c\export_30000.ply")) { Write-Host "[$n] export_30000.ply 없음, 건너뜀" -ForegroundColor Yellow; continue }
    if (-not (Test-Path "$c\nobd_clean.ply")) {
        Write-Host "===== [$n] 부유물 정리 $(Get-Date)"
        python "$Here\clean_section.py" $data "$c\export_30000.ply" "$c\nobd_clean.ply" "$c\nobd_removed.ply"
    }
    if ((Test-Path "$c\nobd_clean.ply") -and -not (Test-Path "$src\$n.ply")) {
        New-Item -ItemType HardLink -Path "$src\$n.ply" -Target "$c\nobd_clean.ply" | Out-Null
    }
}

Write-Host "===== place.py $(Get-Date)"
python "$AVT\Tools\splat_processing\v4\place.py" "src=$src" "out=$out" "tag=_fill"
if (Test-Path "$out\merged_v4_fill.ply") {
    Copy-Item "$out\merged_v4_fill.ply" "$AVT\SplatSource\merged_v4_fill.ply" -Force
    Write-Host "완료: $AVT\SplatSource\merged_v4_fill.ply  -> Unity 에서 Tools > Student Hall > 6 실행" -ForegroundColor Cyan
} else { Write-Host "place.py 결과가 없음 - 위 오류 확인" -ForegroundColor Red }
Stop-Transcript
