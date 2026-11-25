# 3단계: 학습 (LichtFeld Studio 또는 Brush), config.ps1 의 $Trainer / $GpuProfile 대로
# 사용법 (PowerShell, 저장소 폴더에서):  .\3_train\train_all.ps1
# 입력: $Work\Nb\data (2단계 결과)   결과: $Work\Nc_<학습기>\  (PLY, 평가 결과)
# 이미 결과 폴더에 PLY 가 있는 구간은 건너뜀.
. "$PSScriptRoot\..\config.ps1"
if (-not $Trainer) { $Trainer = "lichtfeld" }
if (-not $GpuProfile) { $GpuProfile = "3060ti" }
if (-not $Iterations) { $Iterations = 30000 }
if (-not $ShDegree) { $ShDegree = 2 }

# 그래픽카드별 학습 해상도(긴 변)·스플랫 상한. 메모리가 부족하면 여기서 낮춤
$profiles = @{
    "3060ti" = @{ Width = 1920; Cap = 1500000; Tile = 1 }   # 8GB
    "5070"   = @{ Width = 2560; Cap = 2000000; Tile = 1 }   # 12GB
}
$p = $profiles[$GpuProfile]
if (-not $p) { throw "알 수 없는 GpuProfile: $GpuProfile (3060ti / 5070)" }

$log = Join-Path $Work "night_log.txt"
function Log($msg) { $msg | Tee-Object -FilePath $log -Append }

foreach ($n in $Sections) {
    $data = Join-Path $Work "${n}b\data"
    $out  = Join-Path $Work "${n}c_$Trainer"
    if (-not (Test-Path (Join-Path $data "sparse\0"))) { Write-Host "[$n] 정합 결과 없음: $data (건너뜀)"; continue }
    if ((Test-Path $out) -and (Get-ChildItem $out -Filter *.ply -Recurse -ErrorAction SilentlyContinue)) {
        Write-Host "[$n] 이미 학습됨 (건너뜀)"; continue
    }
    New-Item -ItemType Directory -Force $out | Out-Null
    Log "[$n] 학습 시작 $(Get-Date) / $Trainer / $GpuProfile (폭 $($p.Width), 상한 $($p.Cap))"

    if ($Trainer -eq "lichtfeld") {
        # MCMC(개수 상한 직접 제어) + PPISP(사진마다 다른 노출·WB 보정) + 배경 변조(어두운 곳이 투명하게 남지 않게)
        # mip 필터는 끔: Unity 셰이더(고정 0.3px 확장)와 필터가 어긋남. 3DGUT(--gut)도 끔.
        # 8장마다 1장은 학습에서 빼고 평가에 씀 (--eval --test-every=8)
        $targs = @("-d", $data, "-o", $out, "--strategy=mcmc", "--max-cap=$($p.Cap)", "--iter=$Iterations",
                  "--sh-degree=$ShDegree", "--max-width=$($p.Width)", "--ppisp", "--bg-modulation",
                  "--eval", "--test-every=8", "--headless")
        if ($p.Tile -gt 1) { $targs += "--tile-mode=$($p.Tile)" }
        & $LichtFeld @targs
    }
    elseif ($Trainer -eq "brush") {
        # (Brush 기본 학습 스텝은 30,000)
        $targs = @($data, "--max-splats", "$($p.Cap)", "--sh-degree", "$ShDegree", "--max-resolution", "$($p.Width)",
                   "--export-every", "$Iterations", "--export-path", $out)
        & $Brush @targs
    }
    else { throw "알 수 없는 Trainer: $Trainer (lichtfeld / brush)" }

    if ($LASTEXITCODE -ne 0) { Log "[$n] 학습 실패 (종료 코드 $LASTEXITCODE)"; continue }
    Log "[$n] 학습 끝 $(Get-Date)"
}
