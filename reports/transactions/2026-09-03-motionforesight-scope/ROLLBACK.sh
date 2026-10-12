#!/usr/bin/env pwsh
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$BackupRoot
)

$ErrorActionPreference = 'Stop'

function Get-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path)
}

function Assert-Within([string]$Path, [string]$Root, [string]$Label) {
    $fullPath = Get-FullPath $Path
    $fullRoot = (Get-FullPath $Root).TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label is outside the intended root: $fullPath"
    }
    return $fullPath
}

$project = Get-FullPath $ProjectRoot
$backup = Get-FullPath $BackupRoot
if (-not (Test-Path -LiteralPath $project -PathType Container)) {
    throw "ProjectRoot does not exist: $project"
}
if (-not (Test-Path -LiteralPath $backup -PathType Container)) {
    throw "BackupRoot does not exist: $backup"
}

$newFilesManifest = Join-Path $backup 'new-files.txt'
if (Test-Path -LiteralPath $newFilesManifest) {
    foreach ($rel in Get-Content -LiteralPath $newFilesManifest -Encoding UTF8) {
        if ([string]::IsNullOrWhiteSpace($rel)) { continue }
        $target = Assert-Within (Join-Path $project $rel) $project 'new file target'
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            Remove-Item -LiteralPath $target -Force
        }
    }
}

$movesManifest = Join-Path $backup 'moved-paths.txt'
if (Test-Path -LiteralPath $movesManifest) {
    $moveRows = @(Get-Content -LiteralPath $movesManifest -Encoding UTF8)
    [array]::Reverse($moveRows)
    foreach ($row in $moveRows) {
        if ([string]::IsNullOrWhiteSpace($row)) { continue }
        $parts = $row.Split('|', 2)
        if ($parts.Count -ne 2) { throw "Invalid moved-path row: $row" }
        $source = Assert-Within (Join-Path $project $parts[0]) $project 'restore destination'
        $archived = Assert-Within (Join-Path $project $parts[1]) $project 'archived source'
        if (Test-Path -LiteralPath $archived) {
            if (Test-Path -LiteralPath $source) {
                throw "Restore destination already exists: $source"
            }
            $parent = Split-Path -Parent $source
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
            Move-Item -LiteralPath $archived -Destination $source
        }
    }
}

$modifiedManifest = Join-Path $backup 'modified-files.txt'
foreach ($rel in Get-Content -LiteralPath $modifiedManifest -Encoding UTF8) {
    if ([string]::IsNullOrWhiteSpace($rel)) { continue }
    $saved = Assert-Within (Join-Path (Join-Path $backup 'files') $rel) $backup 'saved original'
    $target = Assert-Within (Join-Path $project $rel) $project 'restore target'
    if (-not (Test-Path -LiteralPath $saved -PathType Leaf)) {
        throw "Saved original is missing: $saved"
    }
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $saved -Destination $target -Force
}

Write-Output 'ROLLBACK_APPLIED=PASS'
