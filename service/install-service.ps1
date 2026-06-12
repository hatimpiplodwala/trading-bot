<#
.SYNOPSIS
    Install the ORB paper-trading bot as a Windows service via NSSM.

.DESCRIPTION
    Registers `python -m ict_bot.main` (the live ORB loop) as an auto-start,
    auto-restart Windows service. Stopping the service sends Ctrl-C, which the
    bot handles gracefully — it finishes the current poll and exits, leaving any
    open position and its resting broker stop untouched.

.PARAMETER NssmPath
    Full path to nssm.exe (download from https://nssm.cc/download).

.PARAMETER ServiceName
    Service name to register (default: ictbot).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File service\install-service.ps1 -NssmPath C:\tools\nssm.exe
#>
param(
    [Parameter(Mandatory = $true)][string]$NssmPath,
    [string]$ServiceName = "ictbot"
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $NssmPath)) { throw "nssm.exe not found at $NssmPath" }
if (-not (Test-Path $python)) { throw "venv python not found at $python - run 'uv sync' first." }
if (-not (Test-Path (Join-Path $repo ".env"))) {
    Write-Warning "No .env at $repo\.env - the bot needs ALPACA_API_KEY / ALPACA_SECRET there."
}

& $NssmPath install $ServiceName $python "-m" "ict_bot.main"
& $NssmPath set $ServiceName AppDirectory $repo
& $NssmPath set $ServiceName AppStopMethodConsole 15000   # send Ctrl-C, wait 15s for a clean stop
& $NssmPath set $ServiceName AppExit Default Restart
& $NssmPath set $ServiceName AppRestartDelay 10000
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName DisplayName "ORB paper-trading bot"
& $NssmPath set $ServiceName Description "Opening Range Breakout live paper loop (SPY, Alpaca paper)."

Write-Host ""
Write-Host "Installed service '$ServiceName'." -ForegroundColor Green
Write-Host "Start:  & '$NssmPath' start $ServiceName"
Write-Host "Stop:   & '$NssmPath' stop $ServiceName    # graceful; leaves the resting broker stop in place"
Write-Host "Logs:   $repo\logs\ictbot.log              # daily rotation, 30 days"
Write-Host "Remove: & '$NssmPath' remove $ServiceName confirm"
