# Stage-2 retrain: composite measure (ground/bucket/airborne) + user-like
# state randomization (brush edits, soil persistence, arm pose variety),
# longer episodes for carry behaviors. Plus a 1d-measure control with the
# same randomization to isolate the composite measure's contribution.

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

$common = "--steps 500000 --skill-dim 4 --randomize-terrain --brush-ops 6 " +
          "--persist-soil-prob 0.5 --arm-random-steps 15 --episode-steps 300 " +
          "--metric euclidean"

$runs = @(
    "scripts/train_tmsd.py $common --measure composite --run-name stage2_comp_s0 --seed 0",
    "scripts/train_tmsd.py $common --measure composite --run-name stage2_comp_s1 --seed 1",
    "scripts/train_tmsd.py $common --measure 1d --run-name stage2_1d_s0 --seed 0"
)

Write-Output ("[{0}] stage-2 wave starting" -f (Get-Date -Format "HH:mm:ss"))
$procs = @()
$i = 0
foreach ($cmd in $runs) {
    $i++
    $procs += Start-Process python -ArgumentList $cmd -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "runs\stage2_proc${i}_out.txt" `
        -RedirectStandardError  "runs\stage2_proc${i}_err.txt"
}
$procs | Wait-Process
Write-Output ("[{0}] stage-2 wave done" -f (Get-Date -Format "HH:mm:ss"))
