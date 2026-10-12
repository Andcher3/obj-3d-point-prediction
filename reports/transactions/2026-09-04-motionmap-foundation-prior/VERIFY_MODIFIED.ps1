param(
 [Parameter(Mandatory=$true)][string]$ProjectRoot,
 [Parameter(Mandatory=$true)][string]$TransactionRoot
)
$ErrorActionPreference='Stop'
$agents=Get-Content -LiteralPath (Join-Path $ProjectRoot 'AGENTS.md') -Raw -Encoding UTF8
$roadmap=Get-Content -LiteralPath (Join-Path $ProjectRoot 'research_roadmap.md') -Raw -Encoding UTF8
$ideaPath=Join-Path $ProjectRoot 'reports\ideas\2026-09-04-motionmap-foundation-prior-redesign.md'
$progressPath=Join-Path $ProjectRoot 'reports\progress\2026-09-04-motionmap-foundation-prior-review.md'
$progressLog=Get-Content -LiteralPath (Join-Path $ProjectRoot 'reports\progress_log.md') -Raw -Encoding UTF8
$draftMarker=$agents.Contains('GROUP_MEETING_DRAFT_NONBINDING=true')
$priorMarker=$roadmap.Contains('CONTINUOUS_FOUNDATION_PRIOR=true')
$speakerNotesSource=$agents.Contains('- `reports/weekly/2026-08-06-flowforesight-method-slides/speaker-notes.txt`')
$newIdea=Test-Path -LiteralPath $ideaPath -PathType Leaf
$newProgress=Test-Path -LiteralPath $progressPath -PathType Leaf
$link=$progressLog.Contains('progress/2026-09-04-motionmap-foundation-prior-review.md')
if(-not $newIdea){throw 'New idea note missing'}
$idea=Get-Content -LiteralPath $ideaPath -Raw -Encoding UTF8
$sources=([regex]::Matches($idea,'(?m)^- (MotionForesight|MotionMap|A Stochastic Conditioning Scheme|MolmoMotion|What Happens Next\?)')).Count
$modifiedCopy=Join-Path $TransactionRoot 'MODIFIED_FILE'
$copyMatches=(Get-FileHash -LiteralPath $modifiedCopy -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath (Join-Path $ProjectRoot 'research_roadmap.md') -Algorithm SHA256).Hash
if(-not $draftMarker -or -not $priorMarker -or $speakerNotesSource -or -not $newProgress -or -not $link -or $sources -ne 5 -or -not $copyMatches){
 throw "Modified verification failed: draft=$draftMarker prior=$priorMarker speaker=$speakerNotesSource idea=$newIdea progress=$newProgress link=$link sources=$sources copy=$copyMatches"
}
Write-Output "MODIFIED_DRAFT_NONBINDING=$draftMarker; CONTINUOUS_FOUNDATION_PRIOR=$priorMarker; SPEAKER_NOTES_SOURCE=$speakerNotesSource; NEW_IDEA=$newIdea; NEW_PROGRESS=$newProgress; PROGRESS_LINK=$link; SOURCES=$sources; MODIFIED_COPY_MATCH=$copyMatches; STATUS=PASS"
