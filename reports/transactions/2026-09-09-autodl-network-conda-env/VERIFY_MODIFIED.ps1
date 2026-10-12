param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$TransactionRoot)
$ErrorActionPreference='Stop'
$log=Join-Path $ProjectRoot 'reports\progress_log.md'
$note=Join-Path $ProjectRoot 'reports\progress\2026-09-09-autodl-network-turbo-conda-environment.md'
$probe=Join-Path $ProjectRoot 'reports\remote\2026-09-09-autodl-network-conda-probe.txt'
$copy=Join-Path $TransactionRoot 'MODIFIED_FILE'
$noteExists=Test-Path -LiteralPath $note -PathType Leaf
$probeExists=Test-Path -LiteralPath $probe -PathType Leaf
$linkExists=Select-String -LiteralPath $log -SimpleMatch 'autodl-network-turbo-conda-environment.md' -Quiet
$githubPass=Select-String -LiteralPath $probe -SimpleMatch 'TURBO_GITHUB_HTTP=200' -Quiet
$hfPass=Select-String -LiteralPath $probe -SimpleMatch 'TURBO_HUGGINGFACE_HTTP=200' -Quiet
$envReady=Select-String -LiteralPath $probe -SimpleMatch 'REMOTE_CONDA_ENV=READY' -Quiet
$pythonMatch=Select-String -LiteralPath $probe -SimpleMatch 'PYTHON_VERSION=3.10.21' -Quiet
$dataBound=Select-String -LiteralPath $probe -SimpleMatch 'ALL_LARGE_PATHS_ON_DATA_DISK=True' -Quiet
$copyMatch=((Get-FileHash -LiteralPath $log -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash)
$pass=$noteExists-and$probeExists-and$linkExists-and$githubPass-and$hfPass-and$envReady-and$pythonMatch-and$dataBound-and$copyMatch
Write-Output ("MODIFIED_NOTE_EXISTS={0}; PROBE_EXISTS={1}; LOG_LINK={2}; TURBO_GITHUB_HTTP_200={3}; TURBO_HF_HTTP_200={4}; ENV_READY={5}; PYTHON_VERSION_MATCH={6}; ALL_LARGE_PATHS_ON_DATA_DISK={7}; MODIFIED_COPY_MATCH={8}; STATUS={9}" -f $noteExists,$probeExists,$linkExists,$githubPass,$hfPass,$envReady,$pythonMatch,$dataBound,$copyMatch,$(if($pass){'PASS'}else{'FAIL'}))
if(-not $pass){exit 1}
