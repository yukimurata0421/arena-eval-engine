param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [string]$UserName = "user",
    [int]$Port = 22,
    [string]$ArenaRoot = "",
    [ValidateSet(3, 5)][int]$IntervalMinutes = 5,
    [int]$MaxDataStaleMinutes = 8,
    [int]$LockStaleMinutes = 4,
    [int]$HealthCheckIntervalMinutes = 30,
    [int]$HealthCheckTimeoutSec = 60,
    [string]$BackgroundTaskName = "ARENA_RPI_SYNC_5MIN",
    [string]$UserPeriodicTaskName = "ARENA_RPI_SYNC_3MIN_USER",
    [string]$UserResumeTaskName = "ARENA_RPI_SYNC_ONRESUME"
)

$ErrorActionPreference = "Stop"

if (-not $ArenaRoot) {
    $ArenaRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$runner = Join-Path $ArenaRoot "scripts\tools\run_rpi_sync_and_health.ps1"
if (-not (Test-Path $runner)) {
    throw "Runner script not found: $runner"
}

$tr = "powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`" -HostName $HostName -UserName $UserName -Port $Port -ArenaRoot `"$ArenaRoot`" -MaxDataStaleMinutes $MaxDataStaleMinutes -LockStaleMinutes $LockStaleMinutes -HealthCheckIntervalMinutes $HealthCheckIntervalMinutes -HealthCheckTimeoutSec $HealthCheckTimeoutSec"

$periodicFull = "\" + $UserPeriodicTaskName
$resumeFull = "\" + $UserResumeTaskName

# User periodic task (every 3 or 5 min while logged on).
& schtasks /Create /TN $periodicFull /SC MINUTE /MO $IntervalMinutes /TR $tr /F /IT | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to create/update periodic task: $periodicFull" }

# User resume task (run once after wake from sleep).
$resumeEvent = "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]"
& schtasks /Create /TN $resumeFull /SC ONEVENT /EC System /MO $resumeEvent /TR $tr /F /IT | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to create/update resume task: $resumeFull" }

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Set-ScheduledTask -TaskName $UserPeriodicTaskName -Settings $settings | Out-Null
Set-ScheduledTask -TaskName $UserResumeTaskName -Settings $settings | Out-Null

& schtasks /Run /TN $periodicFull | Out-Null

$bgFull = "\" + $BackgroundTaskName
& schtasks /Query /TN $bgFull /FO LIST /V > $null 2>&1
$hasBackground = ($LASTEXITCODE -eq 0)
if ($hasBackground) {
    & schtasks /Change /TN $bgFull /ENABLE | Out-Null
    Write-Host "[INFO] Background task is enabled: $bgFull"
}
else {
    Write-Warning "Background task not found: $bgFull"
}

Write-Host "[OK] Installed sync automation"
Write-Host "  - periodic: $periodicFull (${IntervalMinutes} min, interactive)"
Write-Host "  - on-resume: $resumeFull (Power-Troubleshooter event 1)"
if ($hasBackground) {
    Write-Host "  - background: $bgFull (existing task, for restart-before-logon coverage)"
}
Write-Host "  command: $tr"
