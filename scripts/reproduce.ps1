param(
    [string]$Python = 'python',
    [string]$Device = 'cpu',
    [string]$LabData = '',
    [string]$TrackEvalRoot = ''
)

$ErrorActionPreference = 'Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $labRoot
$env:PYTHONIOENCODING = 'utf-8'
if (!$LabData) { $LabData = Join-Path $labRoot 'data' }
if (!$TrackEvalRoot) { $TrackEvalRoot = Join-Path $labRoot 'TrackEval' }
$env:LAB_DATA = $LabData

& $Python scripts/check_data.py --lab-data-root $LabData
if ($LASTEXITCODE -ne 0) { throw 'Kiểm tra dữ liệu thất bại.' }
$labChoices = Get-Content -LiteralPath 'submission/selection.json' -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($labChoice in $labChoices) {
    & $Python scripts/run_tracking.py --source (Join-Path $LabData "$($labChoice.video)/img1") --seq-name $labChoice.video --tracker $labChoice.tracker --conf $labChoice.conf --iou $labChoice.iou --device $Device --out runs/nop_bai --save-video
    if ($LASTEXITCODE -ne 0) { throw "Chạy $($labChoice.video) thất bại." }
}
New-Item -ItemType Directory -Path submission/results -Force | Out-Null
Copy-Item -Path runs/nop_bai/*.txt,runs/nop_bai/*.json -Destination submission/results
& $Python scripts/evaluate_practice.py --trackeval-root $TrackEvalRoot --lab-data-root $LabData --submission runs/nop_bai/video_1.txt --run-name than_final
if ($LASTEXITCODE -ne 0) { throw 'Chấm video_1 thất bại.' }
& $Python scripts/validate_submission.py --lab-data-root $LabData
if ($LASTEXITCODE -ne 0) { throw 'Kiểm tra bài nộp thất bại.' }
