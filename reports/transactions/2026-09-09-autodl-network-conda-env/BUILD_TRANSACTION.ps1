$ErrorActionPreference='Stop'
$root='D:\Pycharm\post_graduate_project'
$tx=Join-Path $root 'reports\transactions\2026-09-09-autodl-network-conda-env'
$backupRoot=Join-Path $tx 'backup\files'
$liveLog=Join-Path $root 'reports\progress_log.md'
$backupLog=Join-Path $backupRoot 'reports\progress_log.md'
$liveNote=Join-Path $root 'reports\progress\2026-09-09-autodl-network-turbo-conda-environment.md'
$liveProbe=Join-Path $root 'reports\remote\2026-09-09-autodl-network-conda-probe.txt'
$empty=Join-Path $tx 'backup\EMPTY'

Copy-Item -LiteralPath $liveLog -Destination (Join-Path $tx 'MODIFIED_FILE') -Force

$diffParts=@()
$part=& git diff --no-index -- $backupLog $liveLog 2>&1
if($LASTEXITCODE -notin @(0,1)){throw "git diff progress_log failed: $LASTEXITCODE"}
$diffParts+=($part -join "`n")
$part=& git diff --no-index -- $empty $liveNote 2>&1
if($LASTEXITCODE -notin @(0,1)){throw "git diff note failed: $LASTEXITCODE"}
$diffParts+=($part -join "`n")
$part=& git diff --no-index -- $empty $liveProbe 2>&1
if($LASTEXITCODE -notin @(0,1)){throw "git diff probe failed: $LASTEXITCODE"}
$diffParts+=($part -join "`n")
Set-Content -LiteralPath (Join-Path $tx 'DIFF_FILE') -Value ($diffParts -join "`n") -Encoding UTF8

& pwsh -NoProfile -ExecutionPolicy Bypass -File (Join-Path $tx 'VERIFY_MODIFIED.ps1') -ProjectRoot $root -TransactionRoot $tx
$modifiedExit=$LASTEXITCODE
Write-Output ('MODIFIED_EXIT='+$modifiedExit)
if($modifiedExit -ne 0){exit $modifiedExit}

$rollbackRoot=Join-Path $tx 'rollback_test'
if(Test-Path -LiteralPath $rollbackRoot){throw "Rollback fixture already exists: $rollbackRoot"}
$fixtureProject=Join-Path $rollbackRoot 'project'
$fixtureTx=Join-Path $rollbackRoot 'transaction'
New-Item -ItemType Directory -Path (Join-Path $fixtureProject 'reports\progress'),(Join-Path $fixtureProject 'reports\remote'),(Join-Path $fixtureTx 'backup\files\reports') -Force | Out-Null
Copy-Item -LiteralPath $liveLog -Destination (Join-Path $fixtureProject 'reports\progress_log.md') -Force
Copy-Item -LiteralPath $liveNote -Destination (Join-Path $fixtureProject 'reports\progress\2026-09-09-autodl-network-turbo-conda-environment.md') -Force
Copy-Item -LiteralPath $liveProbe -Destination (Join-Path $fixtureProject 'reports\remote\2026-09-09-autodl-network-conda-probe.txt') -Force
Copy-Item -LiteralPath (Join-Path $tx 'new-files.txt'),(Join-Path $tx 'modified-files.txt') -Destination $fixtureTx -Force
Copy-Item -LiteralPath $backupLog -Destination (Join-Path $fixtureTx 'backup\files\reports\progress_log.md') -Force

$rollbackScript=[scriptblock]::Create((Get-Content -LiteralPath (Join-Path $tx 'ROLLBACK.sh') -Raw -Encoding UTF8))
$rollbackOutput=& $rollbackScript -ProjectRoot $fixtureProject -TransactionRoot $fixtureTx
$rollbackExit=if($?){0}else{1}
$rollbackOutput | ForEach-Object { Write-Output $_ }
$originalHash=(Get-FileHash -LiteralPath $backupLog -Algorithm SHA256).Hash
$restoredHash=(Get-FileHash -LiteralPath (Join-Path $fixtureProject 'reports\progress_log.md') -Algorithm SHA256).Hash
$restoredHashPass=($restoredHash -eq $originalHash)
$linkAfter=Select-String -LiteralPath (Join-Path $fixtureProject 'reports\progress_log.md') -SimpleMatch 'autodl-network-turbo-conda-environment.md' -Quiet
$noteAfter=Test-Path -LiteralPath (Join-Path $fixtureProject 'reports\progress\2026-09-09-autodl-network-turbo-conda-environment.md') -PathType Leaf
$probeAfter=Test-Path -LiteralPath (Join-Path $fixtureProject 'reports\remote\2026-09-09-autodl-network-conda-probe.txt') -PathType Leaf
$rollbackPass=($rollbackExit -eq 0)-and$restoredHashPass-and(-not $linkAfter)-and(-not $noteAfter)-and(-not $probeAfter)
Write-Output ("ROLLBACK_TEST={0}; RESTORED_LOG_HASH={1}; RESTORED_LINK={2}; NEW_NOTE_REMOVED={3}; NEW_PROBE_REMOVED={4}; EXIT={5}" -f $(if($rollbackPass){'PASS'}else{'FAIL'}),$restoredHashPass,$linkAfter,(-not $noteAfter),(-not $probeAfter),$rollbackExit)
if(-not $rollbackPass){exit 1}

$modifiedHash=(Get-FileHash -LiteralPath (Join-Path $tx 'MODIFIED_FILE') -Algorithm SHA256).Hash
$diffHash=(Get-FileHash -LiteralPath (Join-Path $tx 'DIFF_FILE') -Algorithm SHA256).Hash
Write-Output ('MODIFIED_SHA256='+$modifiedHash)
Write-Output ('DIFF_SHA256='+$diffHash)
