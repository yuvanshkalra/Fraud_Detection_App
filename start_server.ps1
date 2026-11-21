# Convenience launcher for the fraud backend on this Windows laptop.
# The backend is Linux-only (cuML/CUDA 13), so it runs in WSL2 + Ubuntu.
# Run from PowerShell:  .\start_server.ps1
$ErrorActionPreference = 'Continue'
$distro = 'Ubuntu-24.04'
$dir    = '/home/yuvan/fraud-app/fraud_server'

Write-Host "Starting fraud backend in WSL2 ($distro) ..."
wsl -d $distro --cd $dir -- ./run.sh

if ($LASTEXITCODE -ne 0) {
    Read-Host "Server exited with code $LASTEXITCODE. Press Enter to close"
}
