# 2단계: COLMAP 정합 + 가장 큰 모델 선택 + undistort
# 사용법 (PowerShell, 저장소 폴더에서):  .\2_register\register_all.ps1
# 결과: $Work\Nb\colmap\  (db.db, sparse\0,1,...)
#       $Work\Nb\data\    (images + sparse\0)  <- Brush 에 이 폴더를 넣음
# 이미 Nb\data\sparse\0 이 있는 구간은 건너뜀. 밤새 돌려 두는 용도.
. "$PSScriptRoot\..\config.ps1"

$log = Join-Path $Work "night_log.txt"
function Log($msg) { $msg | Tee-Object -FilePath $log -Append }

function Run-Colmap {
    param([string[]]$ColmapArgs)
    & $Colmap @ColmapArgs
    if ($LASTEXITCODE -ne 0) { throw "COLMAP 실패: $($ColmapArgs[0])" }
}

foreach ($n in $Sections) {
    $img  = Join-Path $Work "${n}a"
    $col  = Join-Path $Work "${n}b\colmap"
    $data = Join-Path $Work "${n}b\data"
    $db   = Join-Path $col "db.db"
    $sp   = Join-Path $col "sparse"

    if (Test-Path (Join-Path $data "sparse\0")) { Write-Host "[$n] 이미 완료 (건너뜀)"; continue }
    if (-not (Test-Path $img)) { Write-Host "[$n] 프레임 폴더 없음: $img (건너뜀)"; continue }

    try {
        Log "[$n] 정합 시작 $(Get-Date)"

        # 프레임이 너무 많으면 일정 간격으로 솎아냄 (빠진 사진은 Na_skip 으로 옮김, 삭제 안 함)
        $files = Get-ChildItem $img -Filter *.jpg | Sort-Object Name
        if ($files.Count -gt $ThinAbove) {
            $step = [math]::Ceiling($files.Count / $ThinTarget)
            $skip = Join-Path $Work "${n}a_skip"
            New-Item -ItemType Directory -Force $skip | Out-Null
            for ($i = 0; $i -lt $files.Count; $i++) {
                if ($i % $step -ne 0) { Move-Item $files[$i].FullName $skip }
            }
            $left = (Get-ChildItem $img -Filter *.jpg).Count
            Log "[$n] 솎아냄: $($files.Count) -> $left"
        }

        New-Item -ItemType Directory -Force $sp | Out-Null

        if (-not (Test-Path $db)) {
            # 같은 카메라로 찍은 영상이라 카메라 1개, 렌즈 왜곡 포함 모델(OPENCV)
            Run-Colmap @("feature_extractor", "--database_path", $db, "--image_path", $img,
                         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV")
            Run-Colmap @("${Matcher}_matcher", "--database_path", $db)
        }

        if (-not (Get-ChildItem $sp -Directory -ErrorAction SilentlyContinue)) {
            Run-Colmap @("mapper", "--database_path", $db, "--image_path", $img, "--output_path", $sp)
        }

        # 모델이 여러 개로 나뉘는 경우가 많음 -> 등록된 사진이 가장 많은 모델을 사용
        $best = $null; $bestCount = -1
        foreach ($m in Get-ChildItem $sp -Directory | Sort-Object Name) {
            Log "[$n] 모델 $($m.Name)"
            $info = (& $Colmap model_analyzer --path $m.FullName 2>&1 | Out-String)
            $info -split "`n" | Where-Object { $_ -match "Registered images|Mean reprojection" } | ForEach-Object { Log $_ }
            if ($info -match "Registered images:\s*(\d+)") {
                $c = [int]$Matches[1]
                if ($c -gt $bestCount) { $bestCount = $c; $best = $m.FullName }
            }
        }
        if (-not $best) { throw "정합 결과 모델 없음" }
        Log "[$n] 사용할 모델: $best ($bestCount 장)"

        Run-Colmap @("image_undistorter", "--image_path", $img, "--input_path", $best,
                     "--output_path", $data, "--output_type", "COLMAP")

        Log "[$n] 정합 끝 $(Get-Date)"
    }
    catch {
        Log "[$n] 오류: $_"
    }
}
