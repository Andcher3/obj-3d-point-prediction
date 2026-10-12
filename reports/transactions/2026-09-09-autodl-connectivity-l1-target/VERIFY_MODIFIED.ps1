param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$TransactionRoot)
$ErrorActionPreference='Stop'
$log=Join-Path $ProjectRoot 'reports\progress_log.md'
$note=Join-Path $ProjectRoot 'reports\progress\2026-09-09-autodl-connectivity-and-l1-test-target.md'
$probe=Join-Path $ProjectRoot 'reports\remote\2026-09-09-autodl-connectivity-probe.txt'
$copy=Join-Path $TransactionRoot 'MODIFIED_FILE'
$noteExists=Test-Path -LiteralPath $note -PathType Leaf
$probeExists=Test-Path -LiteralPath $probe -PathType Leaf
$linkExists=Select-String -LiteralPath $log -SimpleMatch 'autodl-connectivity-and-l1-test-target.md' -Quiet
$sshPass=Select-String -LiteralPath $probe -SimpleMatch 'SSH_AUTH=PASS' -Quiet
$logoutPass=Select-String -LiteralPath $probe -SimpleMatch 'SSH_LOGOUT=PASS' -Quiet
$l1Target=Select-String -LiteralPath $note -SimpleMatch '验证 FUTURE-L1 的 latent 扰动能否产生有效的未来事件分支' -Quiet
$copyMatch=((Get-FileHash -LiteralPath $log -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash)
$pass=$noteExists-and$probeExists-and$linkExists-and$sshPass-and$logoutPass-and$l1Target-and$copyMatch
Write-Output ("MODIFIED_NOTE_EXISTS={0}; PROBE_EXISTS={1}; LOG_LINK={2}; SSH_AUTH_PASS={3}; SSH_LOGOUT_PASS={4}; L1_TARGET_RECORDED={5}; MODIFIED_COPY_MATCH={6}; STATUS={7}" -f $noteExists,$probeExists,$linkExists,$sshPass,$logoutPass,$l1Target,$copyMatch,$(if($pass){'PASS'}else{'FAIL'}))
if(-not $pass){exit 1}
