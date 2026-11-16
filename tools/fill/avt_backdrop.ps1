# 4차 보강 스캔에 맞춘 배경 메쉬 + 공간별 자르기 -> Unity
#   avt_fill.ps1 이 만든 avt_out\placed_fill_1..6.ply 를 씀 (place.py 결과)
#   결과: AVT\Assets\Backdrop\v4_fill (공간 목록 + 구운 텍스처), AVT\SplatSource\merged_v4_fill_bd.ply
#   그다음 Unity 에서 Tools > Student Hall > 6 (처음 한 번 가져오기 ~5분)
# 사용: powershell -ExecutionPolicy Bypass -File F:\Sample\tools\fill\avt_backdrop.ps1  (또는 내용 붙여넣기)
$W    = "F:\Sample\4차"
$Here = "F:\Sample\tools\fill"
$AVT  = "E:\Unity\Projects\AVT"
python "$Here\avt_backdrop.py" "src=$W\3_학습결과\avt_out" "cs=$AVT\Assets\Editor\StudentHallBackdrop.cs" "out=$AVT\Assets\Backdrop\v4_fill" "ply=$AVT\SplatSource\merged_v4_fill_bd.ply"
if (Test-Path "$AVT\SplatSource\merged_v4_fill_bd.ply") { Write-Host "완료 -> Unity: Tools > Student Hall > 6" -ForegroundColor Cyan }
else { Write-Host "실패 - 위 오류 확인" -ForegroundColor Red }
