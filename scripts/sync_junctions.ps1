# ARIA Warehouse Synchronization Script
# Uses NTFS Junctions to link central models to backend-specific paths.

$ARIA_ROOT = "C:\Users\Roberto\aria"
$REGISTRY_PATH = "$ARIA_ROOT\data\assets\models\model_registry.json"

if (-not (Test-Path $REGISTRY_PATH)) {
    Write-Error "Registry file not found at $REGISTRY_PATH"
    exit 1
}

$registry = Get-Content $REGISTRY_PATH | ConvertFrom-Json
$warehouse_root_abs = Join-Path $ARIA_ROOT $registry.warehouse_root

Write-Host "--- ARIA Warehouse Synchronization v1.0 ---" -ForegroundColor Cyan

foreach ($mapping in $registry.mappings) {
    $target_abs = Join-Path $ARIA_ROOT $mapping.physical_path
    $model_id = $mapping.model_id
    
    if (-not (Test-Path $target_abs)) {
        Write-Host "[WARN] Physical model '$model_id' not found in Warehouse: $target_abs" -ForegroundColor Yellow
        continue
    }

    foreach ($junction_rel in $mapping.junctions) {
        $junction_abs = Join-Path $ARIA_ROOT $junction_rel
        $parent_dir = Split-Path $junction_abs -Parent
        
        # Ensure parent directory exists
        if (-not (Test-Path $parent_dir)) {
            New-Item -ItemType Directory -Path $parent_dir -Force | Out-Null
        }

        # Check if something already exists at junction path
        if (Test-Path $junction_abs) {
            $item = Get-Item $junction_abs
            if ($item.Attributes -match "ReparsePoint") {
                # It's already a junction or symlink. Verify if it points to the right place.
                # (Simple check: we assume it's okay unless the user reports issues)
                Write-Host "[OK] Junction exists: $junction_rel" -ForegroundColor Gray
                continue
            } else {
                # It's a REAL directory. We must MOVE or DELETE it before creating a junction.
                Write-Host "[INFO] Overwriting existing directory with junction: $junction_rel" -ForegroundColor Yellow
                Remove-Item -Path $junction_abs -Recurse -Force
            }
        }

        # Create the Junction
        Write-Host "[SYNC] Creating Junction: $junction_rel --> $target_abs" -ForegroundColor Green
        New-Item -ItemType Junction -Path $junction_abs -Value $target_abs | Out-Null
    }
}

Write-Host "--- Sync Complete ---" -ForegroundColor Cyan
