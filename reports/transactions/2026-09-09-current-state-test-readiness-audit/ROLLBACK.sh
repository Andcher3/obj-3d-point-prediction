#!/usr/bin/env pwsh
param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$TransactionRoot
)
$ErrorActionPreference = 'Stop'
function Get-FullPath([string]$Path) { [System.IO.Path]::GetFullPath($Path) }
function Assert-Within([string]$Path,[string]$Root,[string]$Label) {
    $fullPath = Get-FullPath $Path
    $fullRoot = (Get-FullPath $Root).TrimEnd('\') + '\'
    if(-not $fullPath.StartsWith($fullRoot,[System.StringComparison]::OrdinalIgnoreCase)){ throw "$Label is outside the intended root: $fullPath" }
    $fullPath
}
$project = Get-FullPath $ProjectRoot
$transaction = Get-FullPath $TransactionRoot
if(-not (Test-Path -LiteralPath $project -PathType Container)){ throw "ProjectRoot does not exist: $project" }
if(-not (Test-Path -LiteralPath $transaction -PathType Container)){ throw "TransactionRoot does not exist: $transaction" }
foreach($rel in Get-Content -LiteralPath (Join-Path $transaction 'new-files.txt') -Encoding UTF8){
    if([string]::IsNullOrWhiteSpace($rel)){ continue }
    $target = Assert-Within (Join-Path $project $rel) $project 'new file target'
    if(Test-Path -LiteralPath $target -PathType Leaf){ Remove-Item -LiteralPath $target -Force }
}
foreach($rel in Get-Content -LiteralPath (Join-Path $transaction 'modified-files.txt') -Encoding UTF8){
    if([string]::IsNullOrWhiteSpace($rel)){ continue }
    $saved = Assert-Within (Join-Path (Join-Path $transaction 'backup\files') $rel) $transaction 'saved original'
    $target = Assert-Within (Join-Path $project $rel) $project 'restore target'
    if(-not (Test-Path -LiteralPath $saved -PathType Leaf)){ throw "Saved original is missing: $saved" }
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $saved -Destination $target -Force
}
Write-Output 'ROLLBACK_APPLIED=PASS'
