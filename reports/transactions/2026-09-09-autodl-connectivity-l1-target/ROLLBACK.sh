#!/usr/bin/env pwsh
param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$TransactionRoot)
$ErrorActionPreference='Stop'
function Full([string]$p){[System.IO.Path]::GetFullPath($p)}
function Within([string]$p,[string]$r,[string]$label){$fp=Full $p;$fr=(Full $r).TrimEnd('\')+'\';if(-not $fp.StartsWith($fr,[System.StringComparison]::OrdinalIgnoreCase)){throw "$label outside intended root: $fp"};$fp}
$project=Full $ProjectRoot;$transaction=Full $TransactionRoot
if(-not(Test-Path -LiteralPath $project -PathType Container)){throw "Missing ProjectRoot: $project"}
if(-not(Test-Path -LiteralPath $transaction -PathType Container)){throw "Missing TransactionRoot: $transaction"}
foreach($rel in Get-Content -LiteralPath (Join-Path $transaction 'new-files.txt') -Encoding UTF8){if([string]::IsNullOrWhiteSpace($rel)){continue};$target=Within (Join-Path $project $rel) $project 'new target';if(Test-Path -LiteralPath $target -PathType Leaf){Remove-Item -LiteralPath $target -Force}}
foreach($rel in Get-Content -LiteralPath (Join-Path $transaction 'modified-files.txt') -Encoding UTF8){if([string]::IsNullOrWhiteSpace($rel)){continue};$saved=Within (Join-Path (Join-Path $transaction 'backup\files') $rel) $transaction 'saved original';$target=Within (Join-Path $project $rel) $project 'restore target';if(-not(Test-Path -LiteralPath $saved -PathType Leaf)){throw "Missing saved original: $saved"};New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null;Copy-Item -LiteralPath $saved -Destination $target -Force}
Write-Output 'ROLLBACK_APPLIED=PASS'
