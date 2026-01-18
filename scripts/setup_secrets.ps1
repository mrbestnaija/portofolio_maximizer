# PowerShell script to setup Docker secrets directory
# Creates secrets directory and sets proper permissions

$ErrorActionPreference = "Stop"

$SECRETS_DIR = "secrets"

Write-Host "🔐 Setting up Docker secrets directory..." -ForegroundColor Cyan

# Create secrets directory if it doesn't exist
if (-not (Test-Path $SECRETS_DIR)) {
    New-Item -ItemType Directory -Path $SECRETS_DIR -Force | Out-Null
    Write-Host "✓ Created secrets directory: $SECRETS_DIR" -ForegroundColor Green
} else {
    Write-Host "✓ Secrets directory already exists: $SECRETS_DIR" -ForegroundColor Green
}

# Set secure permissions (owner read/write/execute only)
$acl = Get-Acl $SECRETS_DIR
$acl.SetAccessRuleProtection($true, $false)
$owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$acl.SetOwner([System.Security.Principal.NTAccount]$owner)
$accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $owner, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
)
$acl.SetAccessRule($accessRule)
Set-Acl $SECRETS_DIR $acl
Write-Host "✓ Set permissions on secrets directory" -ForegroundColor Green

# Create placeholder secret files if they don't exist
$alphaFile = Join-Path $SECRETS_DIR "alpha_vantage_api_key.txt"
if (-not (Test-Path $alphaFile)) {
    "# Placeholder - Replace with your actual Alpha Vantage API key" | Out-File -FilePath $alphaFile -Encoding UTF8
    Write-Host "⚠ Created placeholder file: $alphaFile" -ForegroundColor Yellow
    Write-Host "   Please update with your actual API key!" -ForegroundColor Yellow
}

$finnhubFile = Join-Path $SECRETS_DIR "finnhub_api_key.txt"
if (-not (Test-Path $finnhubFile)) {
    "# Placeholder - Replace with your actual Finnhub API key" | Out-File -FilePath $finnhubFile -Encoding UTF8
    Write-Host "⚠ Created placeholder file: $finnhubFile" -ForegroundColor Yellow
    Write-Host "   Please update with your actual API key!" -ForegroundColor Yellow
}

# Set secure permissions on secret files (owner read/write only)
Get-ChildItem -Path $SECRETS_DIR -Filter "*.txt" | ForEach-Object {
    $acl = Get-Acl $_.FullName
    $acl.SetAccessRuleProtection($true, $false)
    $owner = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $acl.SetOwner([System.Security.Principal.NTAccount]$owner)
    $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $owner, "FullControl", "Allow"
    )
    $acl.SetAccessRule($accessRule)
    Set-Acl $_.FullName $acl
}
Write-Host "✓ Set permissions on secret files" -ForegroundColor Green

Write-Host ""
Write-Host "✅ Secrets directory setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Edit $SECRETS_DIR\alpha_vantage_api_key.txt with your Alpha Vantage API key"
Write-Host "   2. Edit $SECRETS_DIR\finnhub_api_key.txt with your Finnhub API key"
Write-Host "   3. Verify files are not tracked by git: git status --ignored | Select-String secrets"
Write-Host ""
Write-Host "⚠️  IMPORTANT: Never commit secrets to git!" -ForegroundColor Red
Write-Host "   The secrets/ directory is already in .gitignore"
