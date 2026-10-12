param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$rootPath=(Resolve-Path -LiteralPath $Root).Path
$archiveRoot=Join-Path $rootPath 'archive\experiments\molmomotion'
$sourceRoot=Join-Path $rootPath 'experiences_results'
$backupRoot=Join-Path $archiveRoot 'transaction\rollback_backups'
$names=@('MolmoMotion-text-types','MolmoMotion-tracker3d-e2e','MolmoMotion-tracker3d-retest')
foreach($p in @($archiveRoot,$sourceRoot,$backupRoot)){
  $full=[System.IO.Path]::GetFullPath($p)
  if(-not $full.StartsWith($rootPath,[System.StringComparison]::OrdinalIgnoreCase)){throw "path outside rollback root: $full"}
}
New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
$restoredDirs=0
foreach($name in $names){
  $src=Join-Path $archiveRoot $name
  $dst=Join-Path $sourceRoot $name
  if(Test-Path -LiteralPath $src){
    if(Test-Path -LiteralPath $dst){throw "rollback destination exists: $dst"}
    Move-Item -LiteralPath $src -Destination $dst
    $restoredDirs++
  }
}
$restoredFiles=0
if(Test-Path -LiteralPath $backupRoot){
  Get-ChildItem -LiteralPath $backupRoot -Recurse -File | ForEach-Object {
    $rel=$_.FullName.Substring($backupRoot.Length+1)
    $dst=Join-Path $rootPath $rel
    $full=[System.IO.Path]::GetFullPath($dst)
    if(-not $full.StartsWith($rootPath,[System.StringComparison]::OrdinalIgnoreCase)){throw "restore target outside root: $full"}
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination $dst -Force
    $restoredFiles++
  }
}
$sourceCount=($names | Where-Object {Test-Path -LiteralPath (Join-Path $sourceRoot $_)}).Count
$archiveCount=($names | Where-Object {Test-Path -LiteralPath (Join-Path $archiveRoot $_)}).Count
Write-Output "ROLLBACK restored_dirs=$restoredDirs restored_files=$restoredFiles source_count=$sourceCount archive_count=$archiveCount result=PASS exit=0"
