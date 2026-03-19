param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [string]$UserName = "user",
    [int]$Port = 22,
    [string]$ArenaRoot = "",
    [int]$SyncRetryCount = 6,
    [int]$SyncRetryWaitSec = 30,
    [int]$MaxDataStaleMinutes = 20,
    [int]$LockStaleMinutes = 20,
    [int]$HealthCheckIntervalMinutes = 30,
    [int]$HealthCheckTimeoutSec = 120,
    [switch]$StrictHealth,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $ArenaRoot) {
    $ArenaRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$env:PYTHONPATH = Join-Path $ArenaRoot "src"
$syncScript = Join-Path $ArenaRoot "scripts\adsb\ops\rpi_log_sync.py"
$logDir = Join-Path $ArenaRoot "logs\tasks"
$logFile = Join-Path $logDir "run_rpi_sync_and_health.log"
$lockFile = Join-Path $logDir "run_rpi_sync_and_health.lock"
$healthStamp = Join-Path $logDir "dist_1m_health_check.last"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-TaskLog {
    param([string]$Message)
    Add-Content -Path $logFile -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Remove-FileWithRetry {
    param(
        [string]$Path,
        [int]$Retries = 8,
        [int]$WaitMs = 250
    )
    if (-not $Path) { return }
    for ($i = 0; $i -lt [Math]::Max(1, $Retries); $i++) {
        if (-not (Test-Path -LiteralPath $Path)) { return }
        try {
            Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
            return
        }
        catch {
            Start-Sleep -Milliseconds $WaitMs
        }
    }
    Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
}

function Remove-StaleCmdTmpFiles {
    param([int]$OlderThanHours = 6)
    $cutoff = (Get-Date).AddHours(-1 * [Math]::Max(1, $OlderThanHours))
    Get-ChildItem -Path $logDir -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "cmd_*.out.tmp" -or
            $_.Name -like "cmd_*.err.tmp"
        } |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object { Remove-FileWithRetry -Path $_.FullName -Retries 4 -WaitMs 150 }
}

function Invoke-CmdWithTimeout {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int]$TimeoutSec = 120
    )
    $ts = Get-Date -Format "yyyyMMddHHmmssfff"
    $token = [Guid]::NewGuid().ToString("N").Substring(0, 8)
    $outTmp = Join-Path $logDir ("cmd_" + $ts + "_" + $token + ".out.tmp")
    $errTmp = Join-Path $logDir ("cmd_" + $ts + "_" + $token + ".err.tmp")
    Remove-FileWithRetry -Path $outTmp
    Remove-FileWithRetry -Path $errTmp

    $proc = $null
    try {
        $proc = Start-Process -FilePath $FilePath -ArgumentList $Arguments -PassThru -NoNewWindow -RedirectStandardOutput $outTmp -RedirectStandardError $errTmp
        $finished = $proc.WaitForExit([Math]::Max(10, $TimeoutSec) * 1000)
        if (-not $finished) {
            try {
                & taskkill /PID $proc.Id /T /F > $null 2>&1
            }
            catch {}
            throw "command timeout (${TimeoutSec}s): $FilePath $($Arguments -join ' ')"
        }

        $stdout = if (Test-Path -LiteralPath $outTmp) { Get-Content -LiteralPath $outTmp -Raw -ErrorAction SilentlyContinue } else { "" }
        $stderr = if (Test-Path -LiteralPath $errTmp) { Get-Content -LiteralPath $errTmp -Raw -ErrorAction SilentlyContinue } else { "" }

        [pscustomobject]@{
            ExitCode = [int]$proc.ExitCode
            StdOut = $stdout
            StdErr = $stderr
        }
    }
    finally {
        if ($proc -and -not $proc.HasExited) {
            try {
                & taskkill /PID $proc.Id /T /F > $null 2>&1
            }
            catch {}
        }
        Remove-FileWithRetry -Path $outTmp
        Remove-FileWithRetry -Path $errTmp
    }
}

# Acquire cross-task lock (prevents 3min and 5min tasks from colliding).
$hasLock = $false
try {
    if (Test-Path -LiteralPath $lockFile) {
        $lockItem = Get-Item -LiteralPath $lockFile -ErrorAction SilentlyContinue
        if ($lockItem) {
            $lockAgeMin = ((Get-Date) - $lockItem.LastWriteTime).TotalMinutes
            if ($lockAgeMin -lt $LockStaleMinutes) {
                Write-TaskLog "skip: lock exists ($([math]::Round($lockAgeMin, 1)) min old)"
                exit 0
            }
            Remove-Item -LiteralPath $lockFile -Force -ErrorAction SilentlyContinue
            Write-TaskLog "stale lock removed ($([math]::Round($lockAgeMin, 1)) min old)"
        }
    }
    New-Item -ItemType File -Path $lockFile -Force | Out-Null
    $hasLock = $true
    Remove-StaleCmdTmpFiles -OlderThanHours 6
}
catch {
    Write-TaskLog "error: lock acquire failed ($($_.Exception.Message))"
    throw
}

Write-TaskLog ("start host={0} user={1} port={2}" -f $HostName, $UserName, $Port)

function Invoke-SSHText {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteCommand
    )
    $args = @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        "-p", "$Port",
        "$UserName@$HostName",
        $RemoteCommand
    )
    $res = Invoke-CmdWithTimeout -FilePath "ssh" -Arguments $args -TimeoutSec 45
    if ($res.ExitCode -ne 0) {
        throw "ssh command failed: $RemoteCommand"
    }
    return ($res.StdOut | Out-String).Trim()
}

$syncArgs = @(
    $syncScript,
    "--host", $HostName,
    "--user", $UserName,
    "--port", "$Port"
)
if ($DryRun) {
    $syncArgs += "--dry-run"
}

try {
    Write-Host "[task] python $($syncArgs -join ' ')"
    $ok = $false
    for ($i = 1; $i -le [Math]::Max(1, $SyncRetryCount); $i++) {
        $syncRes = Invoke-CmdWithTimeout -FilePath "python" -Arguments $syncArgs -TimeoutSec 180
        if ($syncRes.ExitCode -eq 0) {
            $ok = $true
            break
        }
        if ($i -lt $SyncRetryCount) {
            Write-Host "[task] sync failed (attempt $i/$SyncRetryCount). retry in $SyncRetryWaitSec sec..."
            Start-Sleep -Seconds $SyncRetryWaitSec
        }
    }
    if (-not $ok) {
        throw "sync-rpi-logs failed after $SyncRetryCount attempts"
    }

    if (-not $DryRun) {
    $remoteBase = "/path/to/adsb_monitor/data/logs"
    $latestPlaoLocal = $null
    $verifyTargets = @(
        @{ Name = "adsb_airspy_decoder_metrics_1m.jsonl"; Local = (Join-Path $ArenaRoot "data\adsb_airspy_decoder_metrics_1m.jsonl"); Remote = "$remoteBase/adsb_airspy_decoder_metrics_1m.jsonl" },
        @{ Name = "adsb_airspy_metrics_1m.jsonl"; Local = (Join-Path $ArenaRoot "data\adsb_airspy_metrics_1m.jsonl"); Remote = "$remoteBase/adsb_airspy_metrics_1m.jsonl" },
        @{ Name = "dist_1m.jsonl"; Local = (Join-Path $ArenaRoot "data\dist_1m.jsonl"); Remote = "$remoteBase/dist_1m.jsonl" },
        @{ Name = "dist_pos_health_1m.jsonl"; Local = (Join-Path $ArenaRoot "data\dist_pos_health_1m.jsonl"); Remote = "$remoteBase/dist_pos_health_1m.jsonl" },
        @{ Name = "dist_signal_stats_1m.jsonl"; Local = (Join-Path $ArenaRoot "data\dist_signal_stats_1m.jsonl"); Remote = "$remoteBase/dist_signal_stats_1m.jsonl" }
    )

    foreach ($t in $verifyTargets) {
        $remoteStat = Invoke-SSHText "stat -c '%s %Y' '$($t.Remote)'"
        $parts = $remoteStat -split '\s+'
        if ($parts.Count -lt 2) {
            throw "unexpected remote stat format for $($t.Name): $remoteStat"
        }
        $remoteSize = [long]$parts[0]
        $remoteEpoch = [long]$parts[1]
        $remoteDt = [DateTimeOffset]::FromUnixTimeSeconds($remoteEpoch).LocalDateTime

        $localExists = Test-Path $t.Local
        $localSize = if ($localExists) { [long](Get-Item $t.Local).Length } else { -1 }
        $localDt = if ($localExists) { (Get-Item $t.Local).LastWriteTime } else { Get-Date "1970-01-01" }

        $needsRepair = (-not $localExists) -or ($localSize -lt $remoteSize) -or ($localDt -lt $remoteDt.AddMinutes(-1))
        if ($needsRepair) {
            Add-Content -Path $logFile -Value ("[{0}] repair {1}: local_size={2} remote_size={3} local_mtime={4} remote_mtime={5}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $t.Name, $localSize, $remoteSize, $localDt.ToString("yyyy-MM-dd HH:mm:ss"), $remoteDt.ToString("yyyy-MM-dd HH:mm:ss"))
            $tmp = "$($t.Local).tmp"
            $scpArgs = @(
                "-q",
                "-P", "$Port",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=3",
                "${UserName}@${HostName}:$($t.Remote)",
                $tmp
            )
            $scpRes = Invoke-CmdWithTimeout -FilePath "scp" -Arguments $scpArgs -TimeoutSec 90
            if ($scpRes.ExitCode -ne 0) {
                throw "scp repair failed for $($t.Name)"
            }
            Move-Item -Force $tmp $t.Local
        }
    }

    # Plao: keep newest pos_*.jsonl in sync even if line-sync task lags.
    $plaoRemoteDir = "/path/to/plao/data/logs/plao_pos"
    $latestPlaoRemote = Invoke-SSHText "ls -1t $plaoRemoteDir/pos_*.jsonl 2>/dev/null | head -n 1"
    if ($latestPlaoRemote) {
        $plaoName = Split-Path -Leaf $latestPlaoRemote
        $plaoLocal = Join-Path $ArenaRoot ("data\plao_pos\" + $plaoName)
        $latestPlaoLocal = $plaoLocal
        $plaoStat = Invoke-SSHText "stat -c '%s %Y' '$latestPlaoRemote'"
        $p = $plaoStat -split '\s+'
        if ($p.Count -lt 2) {
            throw "unexpected plao stat format for ${plaoName}: $plaoStat"
        }
        $plaoRemoteSize = [long]$p[0]
        $plaoRemoteEpoch = [long]$p[1]
        $plaoRemoteDt = [DateTimeOffset]::FromUnixTimeSeconds($plaoRemoteEpoch).LocalDateTime

        $plaoExists = Test-Path $plaoLocal
        $plaoLocalSize = if ($plaoExists) { [long](Get-Item $plaoLocal).Length } else { -1 }
        $plaoLocalDt = if ($plaoExists) { (Get-Item $plaoLocal).LastWriteTime } else { Get-Date "1970-01-01" }

        $plaoNeedsRepair = (-not $plaoExists) -or ($plaoLocalSize -lt $plaoRemoteSize) -or ($plaoLocalDt -lt $plaoRemoteDt.AddMinutes(-1))
        if ($plaoNeedsRepair) {
            Add-Content -Path $logFile -Value ("[{0}] repair {1}: local_size={2} remote_size={3} local_mtime={4} remote_mtime={5}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $plaoName, $plaoLocalSize, $plaoRemoteSize, $plaoLocalDt.ToString("yyyy-MM-dd HH:mm:ss"), $plaoRemoteDt.ToString("yyyy-MM-dd HH:mm:ss"))
            $plaoTmp = "$plaoLocal.tmp"
            $scpPlaoArgs = @(
                "-q",
                "-P", "$Port",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=3",
                "${UserName}@${HostName}:$latestPlaoRemote",
                $plaoTmp
            )
            $scpPlaoRes = Invoke-CmdWithTimeout -FilePath "scp" -Arguments $scpPlaoArgs -TimeoutSec 90
            if ($scpPlaoRes.ExitCode -ne 0) {
                throw "scp repair failed for $plaoName"
            }
            Move-Item -Force $plaoTmp $plaoLocal
        }
    }

    $freshnessScript = @'
import json
import pathlib
import sys
import time

max_age_sec = int(sys.argv[1]) * 60
now = time.time()
stale = []

for raw_path in sys.argv[2:]:
    p = pathlib.Path(raw_path)
    try:
        with p.open("rb") as f:
            try:
                f.seek(-8192, 2)
            except OSError:
                f.seek(0)
            lines = f.read().decode("utf-8", "replace").strip().splitlines()
        if not lines:
            stale.append((str(p), "empty"))
            continue
        obj = json.loads(lines[-1])
        ts = float(obj.get("ts"))
        age = int(now - ts)
        if age > max_age_sec:
            stale.append((str(p), f"{age}s"))
    except Exception as e:
        stale.append((str(p), f"error:{e}"))

if stale:
    for path, reason in stale:
        print(f"STALE\t{path}\t{reason}")
    sys.exit(1)

print("OK")
'@
    $freshFiles = @(
        (Join-Path $ArenaRoot "data\adsb_airspy_decoder_metrics_1m.jsonl"),
        (Join-Path $ArenaRoot "data\adsb_airspy_metrics_1m.jsonl"),
        (Join-Path $ArenaRoot "data\dist_1m.jsonl"),
        (Join-Path $ArenaRoot "data\dist_pos_health_1m.jsonl"),
        (Join-Path $ArenaRoot "data\dist_signal_stats_1m.jsonl")
    )
    if ($latestPlaoLocal -and (Test-Path $latestPlaoLocal)) {
        $freshFiles += $latestPlaoLocal
    }
    $pyArgs = @("-", "$MaxDataStaleMinutes") + $freshFiles
        $freshOutput = $freshnessScript | python @pyArgs
        if ($LASTEXITCODE -ne 0) {
            Add-Content -Path $logFile -Value ("[{0}] freshness check failed: {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), ($freshOutput -join " | "))
            Write-Warning "freshness check failed once. retrying sync once more..."
            $syncRetryRes = Invoke-CmdWithTimeout -FilePath "python" -Arguments $syncArgs -TimeoutSec 180
            if ($syncRetryRes.ExitCode -ne 0) {
                throw "second sync attempt failed after freshness check failure"
            }
        $freshOutput = $freshnessScript | python @pyArgs
        if ($LASTEXITCODE -ne 0) {
            throw "freshness check failed after retry: $($freshOutput -join ' | ')"
        }
    }
        Write-TaskLog ("freshness check ok (<= {0} min)" -f $MaxDataStaleMinutes)

        $runHealth = $true
        if (Test-Path $healthStamp) {
            $minsSinceHealth = ((Get-Date) - (Get-Item $healthStamp).LastWriteTime).TotalMinutes
            if ($minsSinceHealth -lt $HealthCheckIntervalMinutes) {
                $runHealth = $false
            }
        }

        if ($runHealth) {
            Write-Host "[task] dist_1m_health_check.py (timeout=${HealthCheckTimeoutSec}s)"
            $healthScript = Join-Path $ArenaRoot "scripts\adsb\ops\dist_1m_health_check.py"
            $outTmp = Join-Path $logDir "dist_1m_health_check.stdout.tmp"
            $errTmp = Join-Path $logDir "dist_1m_health_check.stderr.tmp"
            Remove-Item -Force $outTmp, $errTmp -ErrorAction SilentlyContinue

            $healthArgs = @(
                $healthScript,
                "--rotation-warn-hours", "0",
                "--rotation-fail-hours", "0"
            )
            $proc = Start-Process -FilePath "python" -ArgumentList $healthArgs -PassThru -NoNewWindow -RedirectStandardOutput $outTmp -RedirectStandardError $errTmp
            $finished = $proc.WaitForExit([Math]::Max(10, $HealthCheckTimeoutSec) * 1000)
            if (-not $finished) {
                try { $proc.Kill() } catch {}
                Write-TaskLog ("health check timeout after {0}s; skipped" -f $HealthCheckTimeoutSec)
                Set-Content -Path $healthStamp -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
            }
            else {
                $proc.Refresh()
                $exitCode = if ($null -ne $proc.ExitCode) { [int]$proc.ExitCode } else { -999 }
                if ($exitCode -ne 0) {
                    $errText = ""
                    if (Test-Path $errTmp) {
                        $errText = (Get-Content $errTmp -Raw -ErrorAction SilentlyContinue)
                    }
                    if ($StrictHealth) {
                        throw "dist_1m_health_check.py failed with exit code ${exitCode}: $errText"
                    }
                    Write-Warning "dist_1m_health_check.py reported failure (collector may be stale), but sync succeeded."
                    Write-TaskLog ("health check failed (exit={0})" -f $exitCode)
                    Set-Content -Path $healthStamp -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                }
                else {
                    Set-Content -Path $healthStamp -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                }
            }
            Remove-Item -Force $outTmp, $errTmp -ErrorAction SilentlyContinue
        }
        else {
            Write-TaskLog ("health check skipped (interval={0} min)" -f $HealthCheckIntervalMinutes)
        }
    }
}
catch {
    Write-TaskLog ("error: " + $_.Exception.Message)
    throw
}

finally {
    Write-Host "[task] done"
    Write-TaskLog "done"
    if ($hasLock -and (Test-Path $lockFile)) {
        Remove-Item -Force $lockFile -ErrorAction SilentlyContinue
    }
}
