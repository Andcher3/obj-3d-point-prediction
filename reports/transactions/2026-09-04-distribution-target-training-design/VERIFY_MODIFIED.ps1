param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$TransactionRoot)
$ErrorActionPreference='Stop'
$roadmap=Get-Content -LiteralPath (Join-Path $ProjectRoot 'research_roadmap.md') -Raw -Encoding UTF8
$mainIdea=Get-Content -LiteralPath (Join-Path $ProjectRoot 'reports\ideas\2026-09-04-motionmap-foundation-prior-redesign.md') -Raw -Encoding UTF8
$detailPath=Join-Path $ProjectRoot 'reports\ideas\2026-09-04-distribution-target-adversarial-training-detail.md'
$progressPath=Join-Path $ProjectRoot 'reports\progress\2026-09-04-distribution-target-adversarial-training.md'
$log=Get-Content -LiteralPath (Join-Path $ProjectRoot 'reports\progress_log.md') -Raw -Encoding UTF8
$marker=$roadmap.Contains('DISTRIBUTION_TARGET_TRAINING=true')
$detail=Test-Path -LiteralPath $detailPath -PathType Leaf
$progress=Test-Path -LiteralPath $progressPath -PathType Leaf
$link=$mainIdea.Contains('reports/ideas/2026-09-04-distribution-target-adversarial-training-detail.md')
$progressLink=$log.Contains('progress/2026-09-04-distribution-target-adversarial-training.md')
if(-not $detail){throw 'Detail note missing'}
$d=Get-Content -LiteralPath $detailPath -Raw -Encoding UTF8
$supportSplit=$d.Contains('## 4. 支持图与概率图分开')
$stochasticRole=$d.Contains('Stochastic Conditioning 主要用于这一层')
$copyMatch=(Get-FileHash -LiteralPath (Join-Path $TransactionRoot 'MODIFIED_FILE') -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath (Join-Path $ProjectRoot 'research_roadmap.md') -Algorithm SHA256).Hash
if(-not $marker -or -not $progress -or -not $link -or -not $progressLink -or -not $supportSplit -or -not $stochasticRole -or -not $copyMatch){throw "Verification failed marker=$marker detail=$detail progress=$progress link=$link progressLink=$progressLink support=$supportSplit stochastic=$stochasticRole copy=$copyMatch"}
Write-Output "DISTRIBUTION_TARGET_TRAINING=$marker; DETAIL_NOTE=$detail; PROGRESS_NOTE=$progress; DETAIL_LINK=$link; PROGRESS_LINK=$progressLink; SUPPORT_PROB_SPLIT=$supportSplit; STOCHASTIC_ROLE=$stochasticRole; MODIFIED_COPY_MATCH=$copyMatch; STATUS=PASS"
