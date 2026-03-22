# ARIA Bootstrap Script - Windows PowerShell
# This script restores all Conda environments and dependencies for ARIA backends.

$AriaRoot = Get-Location
$MinicondaRoot = "$HOME\miniconda3"
$CondaExe = "$MinicondaRoot\Scripts\conda.exe"

if (-not (Test-Path $CondaExe)) {
    Write-Error "Miniconda non trovato in $MinicondaRoot. Installa Miniconda prima di continuare."
    exit 1
}

$ManifestPath = Join-Path $AriaRoot "aria_node_controller\config\backends_manifest.json"
if (-not (Test-Path $ManifestPath)) {
    Write-Error "Manifest non trovato in $ManifestPath"
    exit 1
}

$Manifest = Get-Content $ManifestPath | ConvertFrom-Json

Write-Host "--- ARIA BOOTSTRAP START ---" -ForegroundColor Cyan

foreach ($BackendName in $Manifest.backends.psobject.properties.Name) {
    $Cfg = $Manifest.backends.$BackendName
    $EnvPath = Join-Path $AriaRoot $Cfg.env_prefix
    
    Write-Host "Verifica backend: $BackendName..." -ForegroundColor Yellow
    
    if (Test-Path $EnvPath) {
        Write-Host "  [OK] Ambiente già presente in $EnvPath" -ForegroundColor Green
    } else {
        Write-Host "  [!] Ambiente mancante. Inizio creazione..." -ForegroundColor Magenta
        
        if ($Cfg.template) {
            $TemplatePath = Join-Path $AriaRoot $Cfg.template
            Write-Host "  Uso template: $TemplatePath"
            & $CondaExe env create --prefix $EnvPath --file $TemplatePath -y
        } elseif ($Cfg.conda_pkg) {
            Write-Host "  Installazione pacchetto Conda: $($Cfg.conda_pkg)"
            # Parse package and channels
            $CondaArgs = @("create", "--prefix", $EnvPath, "-y")
            $CondaArgs += $Cfg.conda_pkg.Split(" ")
            & $CondaExe $CondaArgs
        } else {
            Write-Warning "  Nessun template o pacchetto definito per $BackendName. Salto."
        }
    }
}

Write-Host "--- ARIA BOOTSTRAP COMPLETATO ---" -ForegroundColor Cyan
Write-Host "Ricorda di riavviare la Tray Icon di ARIA."
