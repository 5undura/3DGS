# 4차 보강 스캔: 스캔 위치 보정 + 벽을 따라가는 배경 면 + 공간별 자르기 -> Unity   (5분쯤)
#   avt_fill.ps1 이 만든 avt_out\placed_fill_1..6.ply 를 씀 (place.py 결과)
#   결과: AVT\Assets\Backdrop\v4_fill (mesh.json + Baked 텍스처), AVT\SplatSource\merged_v4_fill_bd.ply
#   그다음 Unity 에서 Tools > Student Hall > 6 (PLY 가져오기 ~5분)
# 사용: powershell -ExecutionPolicy Bypass -File F:\Sample\tools\fill\avt_backdrop.ps1  (또는 내용 붙여넣기)
$W    = "F:\Sample\4차"
$Here = "F:\Sample\tools\fill"
$AVT  = "E:\Unity\Projects\AVT"
$Ply  = "$AVT\SplatSource\merged_v4_fill_bd.ply"
$before = if (Test-Path $Ply) { (Get-Item $Ply).LastWriteTime } else { $null }
python "$Here\avt_backdrop.py" "src=$W\3_학습결과\avt_out" "cs=$AVT\Assets\Editor\StudentHallBackdrop.cs" "out=$AVT\Assets\Backdrop\v4_fill" "ply=$Ply"
if ((Test-Path $Ply) -and ((Get-Item $Ply).LastWriteTime -ne $before) -and (Test-Path "$AVT\Assets\Backdrop\v4_fill\mesh.json")) {
    Write-Host "완료 -> Unity: Tools > Student Hall > 6" -ForegroundColor Cyan
} else { Write-Host "실패 - 위 오류 확인" -ForegroundColor Red }
