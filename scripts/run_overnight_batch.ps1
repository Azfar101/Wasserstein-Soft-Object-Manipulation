# Overnight batch: 2D-measure trio + seed replication (11 runs, 3 waves of <=4).
# Each wave runs 4 training processes in parallel (probe: ~31.5 sps each).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_overnight_batch.ps1

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

$common = "--steps 200000 --skill-dim 4 --randomize-terrain"

$waves = @(
    @(  # wave 1: the 2D-measure trio (metric claim, richer manifold) + first seed rep
        "scripts/train_tmsd.py $common --measure 2d --metric sliced_w2 --run-name tmsd_sw2_2d_s0 --seed 0",
        "scripts/train_tmsd.py $common --measure 2d --metric euclidean --run-name abl_euclid_2d_s0 --seed 0",
        "scripts/train_tmsd.py $common --measure 2d --metric temporal  --run-name abl_temporal_2d_s0 --seed 0",
        "scripts/train_tmsd.py $common --metric w2 --run-name tmsd_w2_rt_d4_s1 --seed 1"
    ),
    @(  # wave 2: seed replication of the 1D metric tie
        "scripts/train_tmsd.py $common --metric w2        --run-name tmsd_w2_rt_d4_s2      --seed 2",
        "scripts/train_tmsd.py $common --metric euclidean --run-name abl_euclid_rt_d4_s1   --seed 1",
        "scripts/train_tmsd.py $common --metric euclidean --run-name abl_euclid_rt_d4_s2   --seed 2",
        "scripts/train_tmsd.py $common --metric temporal  --run-name abl_temporal_rt_d4_s1 --seed 1"
    ),
    @(  # wave 3: temporal seed + full-state baseline seeds (factorization claim)
        "scripts/train_tmsd.py $common --metric temporal --run-name abl_temporal_rt_d4_s2 --seed 2",
        "scripts/train_tmsd.py $common --metric temporal --phi-input obs --run-name abl_fullstate_temporal_s1 --seed 1",
        "scripts/train_tmsd.py $common --metric temporal --phi-input obs --run-name abl_fullstate_temporal_s2 --seed 2"
    )
)

$w = 0
foreach ($wave in $waves) {
    $w++
    $stamp = Get-Date -Format "HH:mm:ss"
    Write-Output "[$stamp] wave $w starting (${($wave.Count)} runs)"
    $procs = @()
    $i = 0
    foreach ($cmd in $wave) {
        $i++
        $procs += Start-Process python -ArgumentList $cmd -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput "runs\wave${w}_proc${i}_out.txt" `
            -RedirectStandardError  "runs\wave${w}_proc${i}_err.txt"
    }
    $procs | Wait-Process
    $stamp = Get-Date -Format "HH:mm:ss"
    Write-Output "[$stamp] wave $w done"
}
Write-Output "batch complete"
