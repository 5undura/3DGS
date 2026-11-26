# 3단계: 학습 (LichtFeld Studio 또는 Brush), config.ps1 의 $Trainer / $GpuProfile 대로
# 사용법 (PowerShell, 저장소 폴더에서):  .\3_train\train_all.ps1
# 입력: $Work\Nb\data (2단계 결과)   결과: $Work\Nc_<학습기>\  (PLY, 평가 결과)
# 이미 결과 폴더에 PLY 가 있는 구간은 건너뜀.
. "$PSScriptRoot\..\config.ps1"
if (-not $Trainer) { $Trainer = "lichtfeld" }
if (-not $GpuProfile) { $GpuProfile = "3060ti" }
if (-not $Iterations) { $Iterations = 30000 }
if (-not $ShDegree) { $ShDegree = 2 }
if (-not $TrainExtraArgs) { $TrainExtraArgs = @() }

# 그래픽카드별 학습 해상도(긴 변)·스플랫 상한. 메모리가 부족하면 여기서 낮춤
$profiles = @{
    "3060ti" = @{ Width = 1920; Cap = 1500000 }   # 8GB
    "5070"   = @{ Width = 2560; Cap = 2000000 }   # 12GB
}
$p = $profiles[$GpuProfile]
if (-not $p) { throw "알 수 없는 GpuProfile: $GpuProfile (3060ti / 5070)" }

New-Item -ItemType Directory -Force $Work | Out-Null
$log = Join-Path $Work "night_log.txt"
function Log($msg) { $msg | Tee-Object -FilePath $log -Append }

foreach ($n in $Sections) {
    $data = Join-Path $Work "${n}b\data"
    $out  = Join-Path $Work "${n}c_$Trainer"
    if (-not (Test-Path -LiteralPath (Join-Path $data "sparse\0"))) { Write-Host "[$n] 정합 결과 없음: $data (건너뜀)"; continue }
    if ((Test-Path -LiteralPath $out) -and (Get-ChildItem -LiteralPath $out -Filter *.ply -Recurse -ErrorAction SilentlyContinue)) {
        Write-Host "[$n] 이미 학습됨 (건너뜀)"; continue
    }
    New-Item -ItemType Directory -Force $out | Out-Null
    Log "[$n] 학습 시작 $(Get-Date) / $Trainer / $GpuProfile (폭 $($p.Width), 상한 $($p.Cap)) / 추가 옵션 $($TrainExtraArgs -join ' ')"

    if ($Trainer -eq "lichtfeld") {
        # MCMC(개수 상한 직접 제어) + PPISP(사진마다 다른 노출·WB 보정) + 배경 변조(어두운 곳이 투명하게 남지 않게)
        # 옵션 이름은 LichtFeld v0.5.4.3 소스(src/core/argument_parser.cpp)로 확인:
        #   --bg-modulation, --tile-mode 는 v0.5.3 부터 없음 -> --bg-mode=modulation
        #   v0.5.4 부터 기본 결과는 project.licht 뿐 -> --export=ply 를 줘야 <out>\splat_<iter>.ply 가 생김
        # mip 필터는 끔: Unity 셰이더(고정 0.3px 확장)와 필터가 어긋남. 3DGUT(--gut)도 끔.
        # 8장마다 1장은 학습에서 빼고 평가에 씀 (--eval --test-every=8)
        $targs = @("-d", $data, "-o", $out, "--strategy=mcmc", "--max-cap=$($p.Cap)", "--iter=$Iterations",
                  "--sh-degree=$ShDegree", "--max-width=$($p.Width)", "--ppisp", "--bg-mode=modulation", "--export=ply",
                  "--eval", "--test-every=8", "--headless")
        $targs += $TrainExtraArgs
        # | Out-Host: 출력을 화면에 그대로 보여 주면서 끝날 때까지 기다림 ($LASTEXITCODE 유지)
        & $LichtFeld @targs | Out-Host
    }
    elseif ($Trainer -eq "brush") {
        # Brush v0.3.0: 총 스텝 --total-steps (기본 30,000). main 버전은 --total-train-iters 로 바뀜
        $targs = @($data, "--max-splats", "$($p.Cap)", "--sh-degree", "$ShDegree", "--max-resolution", "$($p.Width)",
                   "--total-steps", "$Iterations", "--export-every", "$Iterations", "--export-path", $out) + $TrainExtraArgs
        & $Brush @targs | Out-Host
    }
    else { throw "알 수 없는 Trainer: $Trainer (lichtfeld / brush)" }

    if ($LASTEXITCODE -ne 0) {
        # 중간 PLY 가 남아 있으면 다음 실행에서 "이미 학습됨"으로 건너뛰므로 옆으로 치움 (삭제 안 함)
        $code = $LASTEXITCODE
        $old = "${n}c_${Trainer}_failed_$(Get-Date -Format yyyyMMdd_HHmmss)"
        Rename-Item -LiteralPath $out -NewName $old
        Log "[$n] 학습 실패 (종료 코드 $code) -> $old"
        continue
    }
    Log "[$n] 학습 끝 $(Get-Date)"
}
