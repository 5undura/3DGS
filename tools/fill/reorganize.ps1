# F:\Sample 정리 v2: 촬영 차수별 폴더로 옮기기 + 학습 중간 저장본·캐시 삭제
#   미리보기:  powershell -ExecutionPolicy Bypass -File F:\Sample\fill\reorganize.ps1
#   실제 실행: powershell -ExecutionPolicy Bypass -File F:\Sample\fill\reorganize.ps1 -Run
# v2: 폴더는 한 번에 통째로만 옮김(실패하면 하나씩 옮기지 않고 멈춰서 알려줌).
#     사진 연결(정션)은 먼저 풀고 마지막에 다시 검. 반쯤 옮겨진 상태(4차 + 6 공존)도 합쳐서 이어감.
param([switch]$Run)
$ErrorActionPreference = "Continue"
$R = "F:\Sample"
$log = Join-Path $R "reorganize_log.txt"
$script:moved = 0; $script:deleted = 0; $script:freed = 0; $script:failed = 0

function Say($m, $c = "Gray") { Write-Host $m -ForegroundColor $c; if ($Run) { try { Add-Content -LiteralPath $log -Value $m -Encoding UTF8 } catch {} } }
function P($rel) { Join-Path $R $rel }

function MoveOne($s, $d) {   # 원자적 이동 (같은 드라이브 안에서 이름만 바뀜)
    Say ("  이동  {0}  ->  {1}" -f $s.Substring($R.Length + 1), $d.Substring($R.Length + 1)); $script:moved++
    if (-not $Run) { return }
    try {
        $p = Split-Path $d -Parent
        if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
        if (Test-Path -LiteralPath $s -PathType Container) { [System.IO.Directory]::Move($s, $d) } else { [System.IO.File]::Move($s, $d) }
    } catch {
        $script:failed++
        Say ("  !! 실패: {0}  ({1})" -f $s, $_.Exception.Message) "Red"
    }
}
function Merge($srel, $drel) {   # 대상이 없으면 통째로 이동, 있으면 안의 항목별로 합침
    $s = P $srel; $d = P $drel
    if (-not (Test-Path -LiteralPath $s)) { return }
    if (-not (Test-Path -LiteralPath $d)) { MoveOne $s $d; return }
    if (Test-Path -LiteralPath $s -PathType Leaf) { Say "  [건너뜀: 이미 있음] $drel" "Yellow"; return }
    foreach ($c in Get-ChildItem -LiteralPath $s -Force) { Merge "$srel\$($c.Name)" "$drel\$($c.Name)" }
    if ($Run -and -not (Get-ChildItem -LiteralPath $s -Force)) { Remove-Item -LiteralPath $s }
}
function Unlink($rel) {   # 사진 연결(정션)만 풀기. 원본 사진은 그대로
    $l = P $rel
    if (-not (Test-Path -LiteralPath $l)) { return }
    $it = Get-Item -LiteralPath $l -Force
    if ($it.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        Say "  연결 해제  $rel"; if ($Run) { cmd /c rmdir "$l" | Out-Null }
    } elseif (-not (Get-ChildItem -LiteralPath $l -Force)) {
        Say "  빈 폴더 삭제  $rel"; if ($Run) { Remove-Item -LiteralPath $l }
    } else { Say "  !! $rel 은 연결이 아니고 내용이 있음 - 그대로 둠" "Red" }
}
function Link($linkRel, $targetRel) {
    $l = P $linkRel; $t = P $targetRel
    if (-not (Test-Path -LiteralPath (Split-Path $l -Parent))) { return }
    if (-not (Test-Path -LiteralPath $t)) { Say "  !! 연결 대상 없음: $targetRel" "Red"; return }
    if (Test-Path -LiteralPath $l) { return }
    Say "  연결  $linkRel  ->  $targetRel"
    if ($Run) { New-Item -ItemType Junction -Path $l -Target $t | Out-Null }
}
function DelFile($rel) {
    $f = P $rel
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { return }
    $sz = (Get-Item -LiteralPath $f).Length
    Say ("  삭제  {0}  ({1:N2} GB)" -f $rel, ($sz / 1GB)); $script:deleted++; $script:freed += $sz
    if ($Run) { Remove-Item -LiteralPath $f -Force }
}
function DelCacheDir($rel) {
    $f = P $rel
    if (-not (Test-Path -LiteralPath $f -PathType Container)) { return }
    Say "  삭제  $rel\ (캐시)"; $script:deleted++
    if ($Run) { Remove-Item -LiteralPath $f -Recurse -Force }
}

if ($Run) { Say "===== 실행 $(Get-Date)" "Cyan" } else { Write-Host "===== 미리보기 (실제로 바꾸려면 -Run) =====" -ForegroundColor Cyan }

Say "[0] 사진 연결(정션) 잠시 풀기" "Cyan"
foreach ($n in 1..6) {
    Unlink "6\b\${n}b\data_fill\images"; Unlink "4차\b\${n}b\data_fill\images"; Unlink "4차\2_정합데이터\${n}b\data_fill\images"
}
foreach ($n in 1..7) { Unlink "4\b${n}_fill\images"; Unlink "3차\4_정합데이터\${n}_fill\images" }

Say "[1] 1차 / 2차" "Cyan"
Merge "1.zip" "1차\1.zip"
Merge "2.zip" "2차\2.zip"

Say "[2] 3차 (원래 폴더 3, 4, 5)" "Cyan"
Merge "3"     "3차\0_소니사진"
Merge "3.zip" "3차\0_소니사진.zip"
foreach ($f in "4.mov", "5_noaudio.mov", "4_sdr.mp4") { Merge "4\$f" "3차\1_영상\$f" }
foreach ($f in "Room1.mp4", "room2.MOV", "Room3.mp4") { Merge "5\$f" "3차\1_영상\방추가촬영_$f" }
foreach ($n in 1..7) { Merge "4\$n" "3차\2_프레임\$n" }
foreach ($d in "4s", "5hq", "r1img") { Merge "4\$d" "3차\2_프레임\$d" }
foreach ($n in 1..7) { Merge "4\c$n" "3차\3_COLMAP\$n" }
Merge "4\c5hq" "3차\3_COLMAP\5hq"
foreach ($n in 1..7) { Merge "4\b$n" "3차\4_정합데이터\$n"; Merge "4\b${n}_fill" "3차\4_정합데이터\${n}_fill" }
Merge "4\b5_old" "3차\4_정합데이터\5_old"
foreach ($f in "1.ply", "2.ply", "3.ply", "4.ply", "5.ply", "5_old.ply", "6.ply", "7.ply") { Merge "5\$f" "3차\5_학습결과\Brush\$f" }
foreach ($f in "1_flatA.ply", "1_flatA_preview.ply", "1_flatB.ply", "1_preview.ply") { Merge "5\$f" "3차\5_학습결과\Brush_flatten실험\$f" }
foreach ($n in 1..7) { Merge "4\c${n}_fill" "3차\5_학습결과\fill\$n" }
foreach ($f in "rsmeta.db", "fill_3rd_log.txt", "clean_log.txt") { Merge "4\$f" "3차\기록\$f" }

Say "[3] 4차 (원래 폴더 6, 일부는 이미 4차 로 옮겨짐)" "Cyan"
Merge "6\a" "4차\a"; Merge "6\b" "4차\b"; Merge "6\c" "4차\c"; Merge "6\experiment1" "4차\experiment1"
if (Test-Path -LiteralPath (P "6")) { foreach ($c in Get-ChildItem -LiteralPath (P "6") -Force -File) { Merge "6\$($c.Name)" "4차\$($c.Name)" } }
Merge "4차\a" "4차\1_프레임"
Merge "4차\b" "4차\2_정합데이터"
Merge "4차\c" "4차\3_학습결과"
Merge "4차\experiment1" "4차\3_학습결과\E설정_scale+growth"
foreach ($n in 0..6) { Merge "4차\$n.MP4" "4차\0_영상\$n.MP4" }
foreach ($f in "night_log.txt", "fill_all_log.txt", "clean_log.txt") { Merge "4차\$f" "4차\기록\$f" }

Say "[4] 도구 / 기타" "Cyan"
foreach ($d in "brush", "colmap", "ffmpeg_tool", "fill") { Merge $d "tools\$d" }
foreach ($f in "flatten.py", "flatten2.py", "pick_best.py", "brush_help.txt") { Merge $f "tools\예전_스크립트\$f" }
Merge "textures" "텍스처"
foreach ($d in "cr1", "ps4") { Merge $d "실험_기타\$d" }
foreach ($f in "untitled.psht", "untitled1.psht", "untitled2.psht", "untitled3.psht", "rsmeta.db") { Merge $f "실험_기타\$f" }

Say "[5] 삭제: 학습 중간 저장본 (최종 export_30000 은 남김)" "Cyan"
foreach ($c in "4차\3_학습결과", "6\c", "4차\c") {
    foreach ($n in 1..6) {
        foreach ($s in "05000", "10000", "15000", "20000", "25000") { DelFile "$c\${n}c\export_$s.ply" }
        foreach ($s in "10000", "20000") { DelFile "$c\${n}c_fill\export_$s.ply" }
        DelFile "$c\${n}c_fill\fill_clean_clean_info.npz"
    }
}
foreach ($n in 1..7) {
    foreach ($c in "3차\5_학습결과\fill\$n", "4\c${n}_fill") {
        foreach ($s in "10000", "20000") { DelFile "$c\export_$s.ply" }
        DelFile "$c\fill_clean_clean_info.npz"
    }
}
Say "[6] 삭제: 캐시" "Cyan"
DelCacheDir "target"; DelCacheDir "__pycache__"
foreach ($c in "4차\3_학습결과", "6\c", "4차\c") { DelCacheDir "$c\1c_fill\target" }

Say "[7] 사진 연결 다시 걸기" "Cyan"
foreach ($n in 1..6) { Link "4차\2_정합데이터\${n}b\data_fill\images" "4차\2_정합데이터\${n}b\data\images" }
foreach ($n in 1..7) { Link "3차\4_정합데이터\${n}_fill\images" "3차\4_정합데이터\$n\images" }

if ($Run) {   # 파일·연결이 하나도 안 남은 옛 폴더만 정리
    foreach ($d in "4", "5", "6") {
        $p = P $d
        if (-not (Test-Path -LiteralPath $p)) { continue }
        $left = Get-ChildItem -LiteralPath $p -Recurse -Force | Where-Object { -not $_.PSIsContainer -or ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) }
        if (-not $left) { Remove-Item -LiteralPath $p -Recurse -Force; Say "  빈 폴더 정리 $d" } else { Say "  $d 폴더에 남은 항목이 있어 그대로 둠" "Yellow" }
    }
}

Say ("===== 이동 {0}개, 삭제 {1}개 ({2:N1} GB 확보), 실패 {3}개" -f $script:moved, $script:deleted, ($script:freed / 1GB), $script:failed) "Cyan"
if ($script:failed -gt 0) { Say "실패한 항목이 있으면: Unity·Brush·SuperSplat·탐색기 창과, 이 폴더 안에서 열어 둔 PowerShell 창을 닫고 다시 실행하세요 (이어서 진행됨)" "Red" }
if (-not $Run) { Write-Host "확인 후 실제 실행:  powershell -ExecutionPolicy Bypass -File F:\Sample\fill\reorganize.ps1 -Run" }
