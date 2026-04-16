# setup_kind.ps1
# Run this script ONCE on any new Windows machine to install kind and kubectl locally.
# Usage: PowerShell -ExecutionPolicy Bypass -File scripts/setup_kind.ps1

$ErrorActionPreference = "Stop"
$KindVersion  = "v0.20.0"
$KindBin      = "$PSScriptRoot\..\kind.exe"   # saved at project root (git-ignored)

Write-Host "=== MLOps Local K8s Setup ===" -ForegroundColor Cyan

# ── 1. Download kind ──────────────────────────────────────────────────────────
if (Test-Path $KindBin) {
    Write-Host "[SKIP] kind.exe already exists at $KindBin"
} else {
    Write-Host "[1/4] Downloading kind $KindVersion for Windows..."
    $url = "https://kind.sigs.k8s.io/dl/$KindVersion/kind-windows-amd64"
    Invoke-WebRequest -Uri $url -OutFile $KindBin
    Write-Host "      Saved to: $KindBin"
}

# Add kind to PATH for this session
$env:PATH = "$((Resolve-Path "$PSScriptRoot\..").Path);$env:PATH"

# Verify
$kindVer = & kind version
Write-Host "      $kindVer" -ForegroundColor Green

# ── 2. Verify kubectl (already bundled with Docker Desktop) ───────────────────
Write-Host "[2/4] Checking kubectl..."
try {
    $kubectlVer = & kubectl version --client --short 2>$null
    Write-Host "      $kubectlVer" -ForegroundColor Green
} catch {
    Write-Warning "kubectl not found. Install Docker Desktop or download from https://dl.k8s.io/release"
    exit 1
}

# ── 3. Create Kind cluster ────────────────────────────────────────────────────
Write-Host "[3/4] Creating Kind cluster 'mlops-cluster'..."
$clusterExists = & kind get clusters | Select-String "mlops-cluster"
if ($clusterExists) {
    Write-Host "      [SKIP] Cluster 'mlops-cluster' already exists." -ForegroundColor Yellow
} else {
    $configPath = "$PSScriptRoot\..\kind-config.yaml"
    & kind create cluster --name mlops-cluster --config $configPath
    Write-Host "      Cluster created successfully." -ForegroundColor Green
}

# ── 4. Switch kubectl context ─────────────────────────────────────────────────
Write-Host "[4/4] Switching kubectl context to kind-mlops-cluster..."
& kubectl config use-context kind-mlops-cluster
& kubectl get nodes

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Cyan
Write-Host "Next steps:"
Write-Host "  1. Build images : docker build -t mlops_frauddetect:api-local ."
Write-Host "  2. Load images  : kind load docker-image mlops_frauddetect:api-local --name mlops-cluster"
Write-Host "  3. Deploy       : kubectl apply -f k8s/"
Write-Host "  4. Access API   : kubectl port-forward svc/fraud-api-service 8000:8000"
Write-Host "  5. Access UI    : kubectl port-forward svc/streamlit-service 8501:8501"
