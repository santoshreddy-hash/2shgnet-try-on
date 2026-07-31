# Wire local Windows dataset + SHGNet-56_final.pth + labels.zip into canonical layout.
# Run from repo root in PowerShell:
#
#   .\scripts\wire_local_windows.ps1
#
# Optional overrides:
#   .\scripts\wire_local_windows.ps1 -Images "D:\path\to\images" -Checkpoint "D:\path\to\SHGNet-56_final.pth"

param(
  [string]$Images = ".\dataset annotated\datasetr annotated",
  [string]$Checkpoint = ".\SHGNet-56_final.pth",
  [string]$LabelsZip = ".\labels.zip"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$py = if (Test-Path .\.venv\Scripts\python.exe) { ".\.venv\Scripts\python.exe" } else { "python" }

# $Images may be the pack root (with images/) or the images folder itself
$packOrImages = $Images
$imgArg = @()
if (Test-Path (Join-Path $Images "images")) {
  $imgArg = @("--pack", $Images)
} elseif (Test-Path $Images) {
  $imgArg = @("--images", $Images)
} else {
  Write-Host "[warn] images path not found: $Images — will still try labels.zip + checkpoint"
  $imgArg = @("--skip-images")
}

& $py .\scripts\wire_local_dataset.py `
  @imgArg `
  --checkpoint $Checkpoint `
  --labels-zip $LabelsZip

Write-Host ""
Write-Host "If readiness says READY:"
Write-Host "  $py -m train.train --device cuda"
