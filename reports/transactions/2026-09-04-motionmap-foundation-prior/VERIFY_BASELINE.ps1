param([Parameter(Mandatory=$true)][string]$FixtureRoot)
$ErrorActionPreference='Stop'
$agents=Get-Content -LiteralPath (Join-Path $FixtureRoot 'AGENTS.md') -Raw -Encoding UTF8
$roadmap=Get-Content -LiteralPath (Join-Path $FixtureRoot 'research_roadmap.md') -Raw -Encoding UTF8
$oldIdea=Get-Content -LiteralPath (Join-Path $FixtureRoot 'reports\ideas\2026-09-02-project-status-flowforesight.md') -Raw -Encoding UTF8
$draftMarker=$agents.Contains('GROUP_MEETING_DRAFT_NONBINDING')
$priorMarker=$roadmap.Contains('CONTINUOUS_FOUNDATION_PRIOR')
$oldNotice=$oldIdea.Contains('2026-09-04 方法状态修正')
if($draftMarker -or $priorMarker -or $oldNotice){throw 'Baseline unexpectedly contains new markers'}
Write-Output "BASELINE_GROUP_MEETING_DRAFT_MARKER=$draftMarker; CONTINUOUS_FOUNDATION_PRIOR=$priorMarker; OLD_SNAPSHOT_NOTICE=$oldNotice; STATUS=PASS"
