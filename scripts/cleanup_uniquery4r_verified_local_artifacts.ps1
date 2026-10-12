param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$taskWorkspace = [System.IO.Path]::GetFullPath('D:\Pycharm\post_graduate_project')
$taskRun = Join-Path $taskWorkspace 'experiences_results\uniquery4r-davis-visual-review-2026-10-09'
$taskFreeBefore = (Get-PSDrive -Name D).Free
$taskRemoved = [System.Collections.Generic.List[object]]::new()

function Remove-TaskFile([string]$Path, [string]$Reason) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $resolved.StartsWith($taskWorkspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Cleanup target outside the verified workspace'
    }
    $size = (Get-Item -LiteralPath $resolved).Length
    if ($Apply) { Remove-Item -LiteralPath $resolved }
    $taskRemoved.Add([pscustomobject]@{path=$resolved; bytes=$size; reason=$Reason; deleted=[bool]$Apply})
}

# This file is an outer release ZIP, not a checkpoint used for inference.
# Keep every extracted checkpoint/config and verify all archived bytes first.
$taskRelease = Join-Path $taskWorkspace '.cache\motionforesight\best.pt'
$taskExtracted = Join-Path $taskWorkspace '.cache\motionforesight\extracted'
$taskVerifiedEntries = [System.Collections.Generic.List[object]]::new()
if (Test-Path -LiteralPath $taskRelease) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $beforeHash = (Get-FileHash -LiteralPath $taskRelease -Algorithm SHA256).Hash
    $zip = [System.IO.Compression.ZipFile]::OpenRead($taskRelease)
    $allMatch = $true
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName.EndsWith('/')) { continue }
            $counterpart = [System.IO.Path]::GetFullPath((Join-Path $taskExtracted $entry.FullName))
            if (-not $counterpart.StartsWith($taskExtracted + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Unsafe release archive path'
            }
            if (-not (Test-Path -LiteralPath $counterpart) -or (Get-Item -LiteralPath $counterpart).Length -ne $entry.Length) {
                $allMatch = $false
                break
            }
            $stream = $entry.Open()
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try { $entryHash = [Convert]::ToHexString($sha.ComputeHash($stream)) }
            finally { $sha.Dispose(); $stream.Dispose() }
            $fileHash = (Get-FileHash -LiteralPath $counterpart -Algorithm SHA256).Hash
            if ($entryHash -ne $fileHash) { $allMatch = $false; break }
            $taskVerifiedEntries.Add([pscustomobject]@{archive_entry=$entry.FullName; retained_file=$counterpart; sha256=$fileHash})
        }
    } finally { $zip.Dispose() }
    if ($allMatch -and $taskVerifiedEntries.Count -eq 3 -and
        $beforeHash -eq (Get-FileHash -LiteralPath $taskRelease -Algorithm SHA256).Hash) {
        Remove-TaskFile $taskRelease 'Release ZIP; all 3 entries SHA256-identical to retained extracted files'
    } else { Write-Output 'Release archive retained: full duplicate verification did not pass' }
}

# Generated smoke-test media are reconstructible; keep the JSON evidence.
$taskSmoke = Join-Path $taskRun 'render-smoke-synthetic-NOT-MODEL'
foreach ($file in Get-ChildItem -LiteralPath $taskSmoke -File) {
    if ($file.Name.StartsWith('SYNTHETIC-NOT-MODEL_') -and $file.Extension -in @('.mp4','.png')) {
        Remove-TaskFile $file.FullName 'Reconstructible synthetic rendering test; summary JSON and code retained'
    }
}

# Transport archive duplicates the retained input tree; require validated
# remote transfer of all 180 files before removing this task-specific bundle.
$taskAuditPath = Join-Path $taskRun 'remote-cpu-checks\approved-input-audit.json'
$taskAudit = Get-Content -LiteralPath $taskAuditPath -Raw | ConvertFrom-Json
$taskBundle = Join-Path $taskRun 'input_bundle.tar.gz'
if ((Test-Path -LiteralPath $taskBundle) -and $taskAudit.approved_input_integrity_passed -eq $true -and
    $taskAudit.verified_file_count -eq 180) {
    Remove-TaskFile $taskBundle 'Temporary SSH input bundle; approved local inputs and all validated remote inputs retained'
}
$taskResult = [pscustomobject]@{
    date='2026-10-09'; applied=[bool]$Apply
    removed=$taskRemoved; verified_archive_entries=$taskVerifiedEntries
    removed_bytes=($taskRemoved | Measure-Object bytes -Sum).Sum
    free_bytes_before=$taskFreeBefore; free_bytes_after=(Get-PSDrive -Name D).Free
    protected='Original RGB/masks, queries, real experimental arrays, extracted model checkpoints, source code and reports'
    davis_original_zip_retained=$true
}
$taskJson = $taskResult | ConvertTo-Json -Depth 8
if ($Apply) {
    [System.IO.File]::WriteAllText((Join-Path $taskRun 'local-cleanup-verified.json'), $taskJson + "`n")
}
Write-Output $taskJson
