# 정리 버전 PLY 를 구글 드라이브로 복사 (Colab 비교 이미지용)
#   fill_clean.ply -> 3_학습결과\4차_Brush_fill_clean\N.ply      (보강 + 바탕층 + 정리)
#   nobd_clean.ply -> 3_학습결과\4차_Brush_fill_clean_nobd\N.ply (보강 + 정리, 바탕층 없음 = 팀원 씬에 넣는 버전)
# 사용: powershell -ExecutionPolicy Bypass -File F:\Sample\tools\fill\to_drive.ps1
$Sections = 1..6          # 복도 1만 비교할 거면 1 로 (업로드 용량: 구간당 약 0.9GB)
$W = "F:\Sample\4차\3_학습결과"
$G = "G:\내 드라이브\3DGS\3_학습결과"
foreach ($pair in @(@("fill_clean.ply", "4차_Brush_fill_clean"), @("nobd_clean.ply", "4차_Brush_fill_clean_nobd"))) {
    $name, $dst = $pair
    New-Item -ItemType Directory -Force "$G\$dst" | Out-Null
    foreach ($n in $Sections) {
        $s = "$W\${n}c_fill\$name"
        if (-not (Test-Path $s)) { Write-Host "[$n] $name 없음 (avt_fill.ps1 먼저 실행?)" -ForegroundColor Yellow; continue }
        if (Test-Path "$G\$dst\$n.ply") { Write-Host "[$n] 이미 있음: $dst\$n.ply"; continue }
        Write-Host "[$n] $name -> $dst\$n.ply"
        Copy-Item $s "$G\$dst\$n.ply"
    }
}
Write-Host "복사 끝. 드라이브 동기화(업로드)가 끝난 뒤 Colab 비교 노트북을 실행하세요." -ForegroundColor Cyan
