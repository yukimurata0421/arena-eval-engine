Option Explicit

Dim shell, fso, scriptDir, arenaRoot, cmd
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
arenaRoot = fso.GetParentFolderName(fso.GetParentFolderName(scriptDir))

' Run sync script fully hidden (0 = hidden window, False = don't wait).
cmd = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File """ & scriptDir & "\run_rpi_sync_and_health.ps1"" -HostName localhost -UserName user -Port 22 -ArenaRoot """ & arenaRoot & """ -MaxDataStaleMinutes 8 -LockStaleMinutes 4 -HealthCheckIntervalMinutes 30 -HealthCheckTimeoutSec 60"
shell.Run cmd, 0, False
