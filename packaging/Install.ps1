$ErrorActionPreference = 'Stop'
$version = '0.4.0'
$source = Join-Path $PSScriptRoot 'CrystaLoop'
$base = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Programs\CrystaLoop'))
$target = [IO.Path]::GetFullPath((Join-Path $base $version))
if (-not $target.StartsWith($base + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid installation path' }
if (-not (Test-Path -LiteralPath (Join-Path $source 'CrystaLoop.exe'))) { throw 'Extract the complete release ZIP before installing.' }
if (Get-Process -Name CrystaLoop -ErrorAction SilentlyContinue) { throw 'Close CrystaLoop before installing.' }
New-Item -ItemType Directory -Force -Path $target | Out-Null
Get-ChildItem -LiteralPath $source | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse -Force }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Uninstall.ps1') -Destination $target -Force
$shell = New-Object -ComObject WScript.Shell
$exe = Join-Path $target 'CrystaLoop.exe'
$links = @((Join-Path ([Environment]::GetFolderPath('Desktop')) 'CrystaLoop.lnk'),
           (Join-Path ([Environment]::GetFolderPath('Programs')) 'CrystaLoop.lnk'))
foreach ($path in $links) {
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $exe
    $link.WorkingDirectory = $target
    $link.IconLocation = "$exe,0"
    $link.Save()
}
$key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrystaLoop'
New-Item -Path $key -Force | Out-Null
New-ItemProperty -Path $key -Name DisplayName -Value 'CrystaLoop' -Force | Out-Null
New-ItemProperty -Path $key -Name DisplayVersion -Value $version -Force | Out-Null
New-ItemProperty -Path $key -Name InstallLocation -Value $target -Force | Out-Null
New-ItemProperty -Path $key -Name DisplayIcon -Value "$exe,0" -Force | Out-Null
$uninstall = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $target 'Uninstall.ps1') + '"'
New-ItemProperty -Path $key -Name UninstallString -Value $uninstall -Force | Out-Null
Write-Host "Installed CrystaLoop $version. Start it from the desktop or Start menu."
Write-Host 'Experiment data is stored separately and retained during uninstall.'
