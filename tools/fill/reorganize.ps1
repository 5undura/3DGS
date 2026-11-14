# F:\Sample 정리: 촬영 차수별 폴더로 옮기기 + 학습 중간 저장본·캐시 삭제
#   미리보기 (아무것도 안 바꿈):  powershell -ExecutionPolicy Bypass -File F:\Sample\fill\reorganize.ps1
#   실제 실행:                    powershell -ExecutionPolicy Bypass -File F:\Sample\fill\reorganize.ps1 -Run
# 같은 드라이브 안에서 옮기기라 순식간에 끝남. 이미 옮겨진 항목은 건너뜀(다시 실행해도 안전).
param([switch]$Run)
$ErrorActionPreference = "Continue"
$R = "F:\Sample"
$log = Join-Path $R "reorganize_log.txt"
$moved = 0; $deleted = 0; $freed = 0; $skipped = 0

function Say($m) { Write-Host $m; if ($Run) { Add-Content -LiteralPath $log -Value $m -Encoding UTF8 } }
function Mv($src, $dst) {
    $s = Join-Path $R $src; $d = Join-Path $R $dst
    if (-not (Test-Path -LiteralPath $s)) { return }
    if (Test-Path -LiteralPath $d) { Say "  [건너뜀: 이미 있음] $dst"; $script:skipped++; return }
    Say "  이동  $src  ->  $dst"; $script:moved++
    if ($Run) {
        $p = Split-Path $d -Parent
        if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
        Move-Item -LiteralPath $s -Destination $d
    }
}
function DelFile($rel) {
    $f = Join-Path $R $rel
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { return }
    $sz = (Get-Item -LiteralPath $f).Length
    Say ("  삭제  {0}  ({1:N2} GB)" -f $rel, ($sz / 1GB)); $script:deleted++; $script:freed += $sz
    if ($Run) { Remove-Item -LiteralPath $f -Force }
}
function DelCacheDir($rel) {   # 캐시 폴더만 (안에 링크 없음)
    $f = Join-Path $R $rel
    if (-not (Test-Path -LiteralPath $f -PathType Container)) { return }
    Say "  삭제  $rel\ (캐시 폴더)"; $script:deleted++
    if ($Run) { Remove-Item -LiteralPath $f -Recurse -Force }
}
function FixJunction($linkRel, $targetRel) {   # 옮긴 뒤 사진 폴더 연결(정션) 다시 걸기
    $l = Join-Path $R $linkRel; $t = Join-Path $R $targetRel
    if (-not (Test-Path -LiteralPath (Split-Path $l -Parent))) { return }
    Say "  연결  $linkRel  ->  $targetRel"
    if ($Run) {
        if (Test-Path -LiteralPath $l) { cmd /c rmdir "$l" | Out-Null }   # 링크만 지움 (원본 사진은 그대로)
        New-Item -ItemType Junction -Path $l -Target $t | Out-Null
    }
}

if ($Run) { Say "===== 실행 $(Get-Date)" } else { Write-Host "===== 미리보기 (실제로 바꾸려면 -Run) =====" }

Say "[1] 1차 / 2차 압축본"
Mv "1.zip" "1차\1.zip"
Mv "2.zip" "2차\2.zip"

Say "[2] 3차 촬영 (원래 폴더 3, 4, 5)"
Mv "3"     "3차\0_소니사진"
Mv "3.zip" "3차\0_소니사진.zip"
foreach ($f in "4.mov", "5_noaudio.mov", "4_sdr.mp4") { Mv "4\$f" "3차\1_영상\$f" }
foreach ($f in "Room1.mp4", "room2.MOV", "Room3.mp4") { Mv "5\$f" "3차\1_영상\방추가촬영_$f" }
foreach ($n in 1..7) { Mv "4\$n" "3차\2_프레임\$n" }
foreach ($d in "4s", "5hq", "r1img") { Mv "4\$d" "3차\2_프레임\$d" }
foreach ($n in 1..7) { Mv "4\c$n" "3차\3_COLMAP\$n" }
Mv "4\c5hq" "3차\3_COLMAP\5hq"
foreach ($n in 1..7) { Mv "4\b$n" "3차\4_정합데이터\$n"; Mv "4\b${n}_fill" "3차\4_정합데이터\${n}_fill" }
Mv "4\b5_old" "3차\4_정합데이터\5_old"
foreach ($f in "1.ply", "2.ply", "3.ply", "4.ply", "5.ply", "5_old.ply", "6.ply", "7.ply") { Mv "5\$f" "3차\5_학습결과\Brush\$f" }
foreach ($f in "1_flatA.ply", "1_flatA_preview.ply", "1_flatB.ply", "1_preview.ply") { Mv "5\$f" "3차\5_학습결과\Brush_flatten실험\$f" }
foreach ($n in 1..7) { Mv "4\c${n}_fill" "3차\5_학습결과\fill\$n" }
foreach ($f in "rsmeta.db", "fill_3rd_log.txt", "clean_log.txt") { Mv "4\$f" "3차\기록\$f" }

Say "[3] 4차 촬영 (원래 폴더 6)"
Mv "6" "4차"
foreach ($n in 0..6) { Mv "4차\$n.MP4" "4차\0_영상\$n.MP4" }
Mv "4차\a" "4차\1_프레임"
Mv "4차\b" "4차\2_정합데이터"
Mv "4차\c" "4차\3_학습결과"
Mv "4차\experiment1" "4차\3_학습결과\E설정_scale+growth"
foreach ($f in "night_log.txt", "fill_all_log.txt", "clean_log.txt") { Mv "4차\$f" "4차\기록\$f" }

Say "[4] 도구 / 기타"
foreach ($d in "brush", "colmap", "ffmpeg_tool", "fill") { Mv $d "tools\$d" }
foreach ($f in "flatten.py", "flatten2.py", "pick_best.py", "brush_help.txt") { Mv $f "tools\예전_스크립트\$f" }
Mv "textures" "텍스처"
foreach ($d in "cr1", "ps4") { Mv $d "실험_기타\$d" }
foreach ($f in "untitled.psht", "untitled1.psht", "untitled2.psht", "untitled3.psht", "rsmeta.db") { Mv $f "실험_기타\$f" }

Say "[5] 삭제: 학습 중간 저장본 (최종 export_30000 은 남김)"
foreach ($n in 1..6) {
    $c  = if ($Run -or (Test-Path "$R\4차")) { "4차\3_학습결과" } else { "6\c" }
    foreach ($s in "05000", "10000", "15000", "20000", "25000") { DelFile "$c\${n}c\export_$s.ply" }
    foreach ($s in "10000", "20000") { DelFile "$c\${n}c_fill\export_$s.ply" }
    DelFile "$c\${n}c_fill\fill_clean_clean_info.npz"
}
foreach ($n in 1..7) {
    $c = if ($Run -or (Test-Path "$R\3차\5_학습결과")) { "3차\5_학습결과\fill\$n" } else { "4\c${n}_fill" }
    foreach ($s in "10000", "20000") { DelFile "$c\export_$s.ply" }
    DelFile "$c\fill_clean_clean_info.npz"
}
Say "[6] 삭제: 캐시"
DelCacheDir "target"
DelCacheDir "__pycache__"
DelCacheDir "4차\3_학습결과\1c_fill\target"
DelCacheDir "6\c\1c_fill\target"

Say "[7] 빈곳채우기 데이터의 사진 연결 다시 걸기"
foreach ($n in 1..6) { FixJunction "4차\2_정합데이터\${n}b\data_fill\images" "4차\2_정합데이터\${n}b\data\images" }
foreach ($n in 1..7) { FixJunction "3차\4_정합데이터\${n}_fill\images" "3차\4_정합데이터\$n\images" }

foreach ($d in "4", "5") {   # 비었으면 빈 폴더만 정리
    $p = Join-Path $R $d
    if ($Run -and (Test-Path -LiteralPath $p) -and -not (Get-ChildItem -LiteralPath $p -Force)) { Remove-Item -LiteralPath $p; Say "  빈 폴더 정리 $d" }
}
Say ("===== 이동 {0}개, 삭제 {1}개 ({2:N1} GB 확보), 건너뜀 {3}개" -f $moved, $deleted, ($freed / 1GB), $skipped)
if (-not $Run) { Write-Host "확인 후 실제 실행:  powershell -ExecutionPolicy Bypass -File F:\Sample\fill\reorganize.ps1 -Run" }
