$ErrorActionPreference = 'Stop'
$base = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Programs\CrystaLoop'))
$target = [IO.Path]::GetFullPath($PSScriptRoot)
if (-not $target.StartsWith($base + '\', [StringComparison]::OrdinalIgnoreCase) -or $target -eq $base) { throw 'Refusing to remove an unrecognized installation path.' }
if (Get-Process -Name CrystaLoop -ErrorAction SilentlyContinue) { throw 'Close CrystaLoop before uninstalling.' }
$shell = New-Object -ComObject WScript.Shell
foreach ($folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    $path = Join-Path $folder 'CrystaLoop.lnk'
    if (Test-Path -LiteralPath $path) {
        if ($shell.CreateShortcut($path).TargetPath -eq (Join-Path $target 'CrystaLoop.exe')) { Remove-Item -LiteralPath $path }
    }
}
$key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrystaLoop'
if ((Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).InstallLocation -eq $target) { Remove-Item -LiteralPath $key -Recurse }
Remove-Item -LiteralPath $target -Recurse -Force
Write-Host 'CrystaLoop removed. Recorded experiments and settings were retained.'
