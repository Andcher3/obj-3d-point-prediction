param([Parameter(Mandatory=$true)][string]$FixtureRoot)
$ErrorActionPreference='Stop'
$roadmap=Get-Content -LiteralPath (Join-Path $FixtureRoot 'research_roadmap.md') -Raw -Encoding UTF8
$idea=Get-Content -LiteralPath (Join-Path $FixtureRoot 'reports\ideas\2026-09-04-motionmap-foundation-prior-redesign.md') -Raw -Encoding UTF8
$a=$roadmap.Contains('DISTRIBUTION_TARGET_TRAINING')
$b=$idea.Contains('distribution-target-adversarial-training-detail.md')
if($a -or $b){throw 'Baseline contains follow-up markers'}
Write-Output "BASELINE_DISTRIBUTION_TARGET_TRAINING=$a; DETAIL_LINK=$b; STATUS=PASS"
