# MI 2x2 completion batch: continuous-DIAYN objective, soil-phi vs full-state phi,
# 3 seeds each. Completes the (objective family) x (conditioning) comparison table.

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

$common = "--steps 200000 --skill-dim 4 --randomize-terrain --objective mi"

$waves = @(
    @(
        "scripts/train_tmsd.py $common --run-name mi_soil_s0 --seed 0",
        "scripts/train_tmsd.py $common --run-name mi_soil_s1 --seed 1",
        "scripts/train_tmsd.py $common --phi-input obs --run-name mi_full_s0 --seed 0",
        "scripts/train_tmsd.py $common --phi-input obs --run-name mi_full_s1 --seed 1"
    ),
    @(
        "scripts/train_tmsd.py $common --run-name mi_soil_s2 --seed 2",
        "scripts/train_tmsd.py $common --phi-input obs --run-name mi_full_s2 --seed 2"
    )
)

$w = 0
foreach ($wave in $waves) {
    $w++
    Write-Output ("[{0}] MI wave $w starting" -f (Get-Date -Format "HH:mm:ss"))
    $procs = @()
    $i = 0
    foreach ($cmd in $wave) {
        $i++
        $procs += Start-Process python -ArgumentList $cmd -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput "runs\mi_wave${w}_proc${i}_out.txt" `
            -RedirectStandardError  "runs\mi_wave${w}_proc${i}_err.txt"
    }
    $procs | Wait-Process
    Write-Output ("[{0}] MI wave $w done" -f (Get-Date -Format "HH:mm:ss"))
}
Write-Output "MI batch complete"
