param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$TransactionRoot
)

$ErrorActionPreference = 'Stop'
$agents = Get-Content -LiteralPath (Join-Path $ProjectRoot 'AGENTS.md') -Raw -Encoding UTF8
$roadmap = Get-Content -LiteralPath (Join-Path $ProjectRoot 'research_roadmap.md') -Raw -Encoding UTF8
$progress = Get-Content -LiteralPath (Join-Path $ProjectRoot 'reports\progress_log.md') -Raw -Encoding UTF8
if (-not $agents.Contains('ACTIVE_RESEARCH_SCOPE=VIDEO_MOTION_FORECASTING')) { throw 'AGENTS marker missing' }
if (-not $roadmap.Contains('ACTIVE_RESEARCH_SCOPE=VIDEO_MOTION_FORECASTING')) { throw 'roadmap marker missing' }
if (-not $roadmap.Contains('p(D_future | V_obs, M_target)')) { throw 'probabilistic task missing' }
if (-not $roadmap.Contains('z_global')) { throw 'semantic/local factorization missing' }

foreach ($rel in @('HAR', 'data', 'src', 'requirements', 'human+activity+recognition+using+smartphones.zip')) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $rel)) { throw "root IMU asset remains: $rel" }
}
foreach ($rel in @('archive\imu\workspace\HAR', 'archive\imu\workspace\data', 'archive\imu\workspace\src', 'archive\imu\workspace\requirements', 'archive\imu\workspace\human+activity+recognition+using+smartphones.zip', 'archive\imu\README.md')) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $rel))) { throw "archive asset missing: $rel" }
}

$activeWeekly = @(Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'reports\weekly') -Directory | Select-Object -ExpandProperty Name | Sort-Object)
$expectedWeekly = @('2026-08-06-flowforesight-method-slides', '2026-08-06-motion-forecast-papers')
if (Compare-Object $activeWeekly $expectedWeekly) { throw 'active weekly folders differ' }

$links = [regex]::Matches($progress, '\]\((progress/[^)]+)\)')
foreach ($match in $links) {
    $rel = $match.Groups[1].Value.Replace('/', '\')
    if (-not (Test-Path -LiteralPath (Join-Path (Join-Path $ProjectRoot 'reports') $rel))) {
        throw "broken progress link: $rel"
    }
}

$rows = Get-Content -LiteralPath (Join-Path $TransactionRoot 'moved-paths.txt') -Encoding UTF8
$moved = 0
foreach ($row in $rows) {
    if ([string]::IsNullOrWhiteSpace($row)) { continue }
    $parts = $row.Split('|', 2)
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $parts[0])) { throw "move source still exists: $($parts[0])" }
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $parts[1]))) { throw "move target missing: $($parts[1])" }
    $moved++
}

Write-Output "MODIFIED_ACTIVE_SCOPE=VIDEO_MOTION_FORECASTING; ROOT_IMU_ASSETS=False; ARCHIVED_IMU_ASSETS=True; MOVED_PATHS=$moved; ACTIVE_WEEKLY=$($activeWeekly.Count); PROGRESS_LINKS=$($links.Count); STATUS=PASS"
