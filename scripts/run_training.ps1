# Trains the two diffusion models one after another (one job at a time, so timings use the whole CPU).
# Usage (from project3):  powershell -ExecutionPolicy Bypass -File scripts\run_training.ps1
# Safe to re-run after an interruption: --resume continues from checkpoints\<run>\model_last.pt.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$log = "results\metrics\run_training.log"
function Step($name, [string[]]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -FilePath $log -Append
    & $py @cmd *>> $log
    if ($LASTEXITCODE -ne 0) { "FAILED $name (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $name" | Tee-Object -FilePath $log -Append
}
Step "train mdlm_small" @("-m", "mdlm.train", "--config", "configs\small.yaml", "--resume")
Step "train mdlm_base" @("-m", "mdlm.train", "--config", "configs\base.yaml", "--resume")
"ALL DONE" | Tee-Object -FilePath $log -Append
