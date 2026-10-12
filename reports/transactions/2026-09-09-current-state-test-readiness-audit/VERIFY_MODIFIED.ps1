param(
 [Parameter(Mandatory=$true)][string]$ProjectRoot,
 [Parameter(Mandatory=$true)][string]$TransactionRoot
)
$ErrorActionPreference='Stop'
$log = Join-Path $ProjectRoot 'reports\progress_log.md'
$note = Join-Path $ProjectRoot 'reports\progress\2026-09-09-current-state-and-test-readiness-audit.md'
$copy = Join-Path $TransactionRoot 'MODIFIED_FILE'
$noteExists = Test-Path -LiteralPath $note -PathType Leaf
$linkExists = Select-String -LiteralPath $log -SimpleMatch 'current-state-and-test-readiness-audit.md' -Quiet
$scope = Select-String -LiteralPath $note -SimpleMatch '任务边界已固定' -Quiet
$route = Select-String -LiteralPath $note -SimpleMatch '候选生成路线暂定' -Quiet
$runtime = Select-String -LiteralPath $note -SimpleMatch 'WinError -2146893795' -Quiet
$copyMatch = ((Get-FileHash -LiteralPath $log -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash)
$pass = $noteExists -and $linkExists -and $scope -and $route -and $runtime -and $copyMatch
Write-Output ("MODIFIED_NOTE_EXISTS={0}; LOG_LINK={1}; SCOPE_RECORDED={2}; ROUTE_RECORDED={3}; RUNTIME_RECORDED={4}; MODIFIED_COPY_MATCH={5}; STATUS={6}" -f $noteExists,$linkExists,$scope,$route,$runtime,$copyMatch,$(if($pass){'PASS'}else{'FAIL'}))
if(-not $pass){ exit 1 }
