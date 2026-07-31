# Wire local Windows dataset + SHGNet-56_final.pth into canonical layout.
# Run from repo root in PowerShell:

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$py = if (Test-Path .\.venv\Scripts\python.exe) { ".\.venv\Scripts\python.exe" } else { "python" }

& $py .\scripts\wire_local_dataset.py `
  --pack ".\dataset annotated\datasetr annotated" `
  --checkpoint ".\SHGNet-56_final.pth"

Write-Host ""
Write-Host "Then train:"
Write-Host "  $py -m train.train --device cuda"
