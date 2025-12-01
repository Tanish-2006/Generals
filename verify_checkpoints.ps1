# Checkpoint Verification Script for Generals RL Training
# Usage: Run this before and after training iterations to verify checkpoint updates

$CHECKPOINT_DIR = "data\checkpoints"
$MODEL_LATEST = Join-Path $CHECKPOINT_DIR "model_latest.pth"
$MODEL_OLD = Join-Path $CHECKPOINT_DIR "model_old.pth"

function Get-FileInfo {
    param($Path)
    
    if (Test-Path $Path) {
        $file = Get-Item $Path
        $hash = (Get-FileHash -Path $Path -Algorithm MD5).Hash
        
        return @{
            Exists = $true
            Size = $file.Length
            Modified = $file.LastWriteTime
            MD5 = $hash
        }
    } else {
        return @{
            Exists = $false
        }
    }
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "CHECKPOINT VERIFICATION - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Check model_latest.pth
Write-Host "MODEL_LATEST.PTH" -ForegroundColor Yellow
$latest_info = Get-FileInfo -Path $MODEL_LATEST
if ($latest_info.Exists) {
    Write-Host "  ✓ Exists" -ForegroundColor Green
    Write-Host "  Size:     $($latest_info.Size) bytes"
    Write-Host "  Modified: $($latest_info.Modified)"
    Write-Host "  MD5:      $($latest_info.MD5)" -ForegroundColor Magenta
} else {
    Write-Host "  ✗ NOT FOUND" -ForegroundColor Red
}
Write-Host ""

# Check model_old.pth
Write-Host "MODEL_OLD.PTH" -ForegroundColor Yellow
$old_info = Get-FileInfo -Path $MODEL_OLD
if ($old_info.Exists) {
    Write-Host "  ✓ Exists" -ForegroundColor Green
    Write-Host "  Size:     $($old_info.Size) bytes"
    Write-Host "  Modified: $($old_info.Modified)"
    Write-Host "  MD5:      $($old_info.MD5)" -ForegroundColor Magenta
} else {
    Write-Host "  ✗ NOT FOUND" -ForegroundColor Red
}
Write-Host ""

# Compare checksums
if ($latest_info.Exists -and $old_info.Exists) {
    Write-Host "COMPARISON" -ForegroundColor Yellow
    if ($latest_info.MD5 -eq $old_info.MD5) {
        Write-Host "  ⚠ WARNING: model_latest and model_old are IDENTICAL" -ForegroundColor Red
        Write-Host "  → This is expected at iteration start, before Arena acceptance" -ForegroundColor Gray
    } else {
        Write-Host "  ✓ Checkpoints are DIFFERENT (expected after training)" -ForegroundColor Green
    }
    
    $time_diff = ($latest_info.Modified - $old_info.Modified).TotalMinutes
    Write-Host "  Time difference: $([math]::Round($time_diff, 2)) minutes"
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Save to log file for tracking
$log_entry = @"
[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]
  model_latest: $($latest_info.MD5) | Modified: $($latest_info.Modified)
  model_old:    $($old_info.MD5) | Modified: $($old_info.Modified)

"@

Add-Content -Path "checkpoint_verification.log" -Value $log_entry
Write-Host "✓ Logged to checkpoint_verification.log" -ForegroundColor Green
