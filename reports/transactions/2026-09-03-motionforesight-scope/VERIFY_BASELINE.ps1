param(
    [Parameter(Mandatory = $true)]
    [string]$FixtureRoot
)

$ErrorActionPreference = 'Stop'
$agents = Get-Content -LiteralPath (Join-Path $FixtureRoot 'AGENTS.md') -Raw -Encoding UTF8
$roadmap = Get-Content -LiteralPath (Join-Path $FixtureRoot 'research_roadmap.md') -Raw -Encoding UTF8
if (-not ($agents.Contains('first two IMU-based topics') -and $roadmap.Contains('手机 IMU'))) {
    throw 'baseline scope assertion failed'
}
Write-Output 'BASELINE_ACTIVE_SCOPE=IMU; ARCHIVE_PRESENT=False; STATUS=PASS'
