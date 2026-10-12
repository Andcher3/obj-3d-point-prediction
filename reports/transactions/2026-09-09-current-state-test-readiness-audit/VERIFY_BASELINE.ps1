param([Parameter(Mandatory=$true)][string]$FixtureRoot)
$ErrorActionPreference='Stop'
$log = Join-Path $FixtureRoot 'reports\progress_log.md'
$note = Join-Path $FixtureRoot 'reports\progress\2026-09-09-current-state-and-test-readiness-audit.md'
$noteExists = Test-Path -LiteralPath $note -PathType Leaf
$linkExists = Select-String -LiteralPath $log -SimpleMatch 'current-state-and-test-readiness-audit.md' -Quiet
$pass = (-not $noteExists) -and (-not $linkExists)
Write-Output ("BASELINE_NOTE_EXISTS={0}; BASELINE_LOG_LINK={1}; STATUS={2}" -f $noteExists,$linkExists,$(if($pass){'PASS'}else{'FAIL'}))
if(-not $pass){ exit 1 }
