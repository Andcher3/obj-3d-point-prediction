param([Parameter(Mandatory=$true)][string]$FixtureRoot)
$ErrorActionPreference='Stop'
$log=Join-Path $FixtureRoot 'reports\progress_log.md'
$note=Join-Path $FixtureRoot 'reports\progress\2026-09-09-autodl-network-turbo-conda-environment.md'
$probe=Join-Path $FixtureRoot 'reports\remote\2026-09-09-autodl-network-conda-probe.txt'
$noteExists=Test-Path -LiteralPath $note -PathType Leaf
$probeExists=Test-Path -LiteralPath $probe -PathType Leaf
$linkExists=Select-String -LiteralPath $log -SimpleMatch 'autodl-network-turbo-conda-environment.md' -Quiet
$pass=(-not $noteExists)-and(-not $probeExists)-and(-not $linkExists)
Write-Output ("BASELINE_NOTE_EXISTS={0}; BASELINE_PROBE_EXISTS={1}; BASELINE_LOG_LINK={2}; STATUS={3}" -f $noteExists,$probeExists,$linkExists,$(if($pass){'PASS'}else{'FAIL'}))
if(-not $pass){exit 1}
