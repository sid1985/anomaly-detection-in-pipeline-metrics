<#
.SYNOPSIS
    One-time bootstrap: creates Azure resources and prints GitHub secrets to set.

.DESCRIPTION
    Run this ONCE before the first GitHub Actions deployment.
    It creates the resource group, deploys the Bicep template (with a placeholder
    image), creates a service principal scoped to the RG + ACR, and outputs all
    the GitHub secrets you need to configure.

.PARAMETER ResourceGroup
    Name of the Azure resource group to create.  Default: pl-paper6-rg

.PARAMETER Location
    Azure region.  Default: eastus

.PARAMETER AcrName
    Container Registry name - must be globally unique, 5-50 alphanumeric chars (no hyphens).
    Default: plpaper6acr

.PARAMETER AppName
    Base name for the Container App and related resources.  Default: pl-paper6

.EXAMPLE
    .\infra\setup.ps1 -AcrName plpaper6acr
#>

[CmdletBinding()]
param(
    [string] $ResourceGroup = 'pl-paper6-rg',
    [string] $Location      = 'eastus',
    [string] $AcrName       = 'plpaper6acr',
    [string] $AppName       = 'pl-paper6'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step([string] $msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]   $msg) { Write-Host "    [ok] $msg"  -ForegroundColor Green }
function Write-Info([string] $msg) { Write-Host "    $msg"       -ForegroundColor Gray }

# -- 1. Azure login -----------------------------------------------------------
Write-Step "Checking Azure login"
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Not logged in - running az login..." -ForegroundColor Yellow
    az login | Out-Null
    $account = az account show | ConvertFrom-Json
}
$subscriptionId = $account.id
$tenantId       = $account.tenantId
Write-Ok "Logged in: $($account.user.name) | Subscription: $subscriptionId"

# -- 2. Create resource group -------------------------------------------------
Write-Step "Creating resource group '$ResourceGroup' in '$Location'"
az group create --name $ResourceGroup --location $Location --output none
Write-Ok "Resource group ready"

# -- 3. Deploy Bicep (initial: placeholder image) -----------------------------
Write-Step "Deploying Bicep infrastructure (first deploy uses placeholder image)"
$bicepFile = Join-Path $PSScriptRoot "main.bicep"

$deployOutput = az deployment group create `
    --resource-group $ResourceGroup `
    --template-file  $bicepFile `
    --parameters     acrName=$AcrName appName=$AppName `
    --output json | ConvertFrom-Json

$acrLoginServer = $deployOutput.properties.outputs.acrLoginServer.value
$acaFqdn        = $deployOutput.properties.outputs.containerAppFqdn.value
$acaName        = $deployOutput.properties.outputs.containerAppName.value
$acaEnvName     = $deployOutput.properties.outputs.caEnvironmentName.value

Write-Ok "ACR login server : $acrLoginServer"
Write-Ok "ACA FQDN         : $acaFqdn"
Write-Ok "ACA app name     : $acaName"
Write-Ok "ACA env name     : $acaEnvName"

# -- 4. Create service principal for GitHub Actions ---------------------------
Write-Step "Creating service principal '$AppName-github-actions'"

$rgScope  = "/subscriptions/$subscriptionId/resourceGroups/$ResourceGroup"
$acrScope = (az acr show --name $AcrName --resource-group $ResourceGroup --query id -o tsv)

# Contributor on the resource group (to run az deployment + az containerapp update)
$spJson = az ad sp create-for-rbac `
    --name        "$AppName-github-actions" `
    --role        "Contributor" `
    --scopes      $rgScope `
    --sdk-auth `
    --output json | ConvertFrom-Json

$spClientId = $spJson.clientId
Write-Ok "Service principal created: $spClientId"

# AcrPush so the workflow can push images
az role assignment create `
    --assignee  $spClientId `
    --role      "AcrPush" `
    --scope     $acrScope `
    --output    none
Write-Ok "Granted AcrPush on ACR"

# -- 5. Print GitHub secrets --------------------------------------------------
$azureCreds = $spJson | ConvertTo-Json -Depth 5 -Compress

Write-Host ("`n" + ("=" * 70)) -ForegroundColor Yellow
Write-Host "  ADD THESE SECRETS TO YOUR GITHUB REPO" -ForegroundColor Yellow
Write-Host "  (Settings > Secrets and variables > Actions > New repository secret)" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor Yellow

$secrets = [ordered]@{
    AZURE_CREDENTIALS    = $azureCreds
    ACR_NAME             = $AcrName
    AZURE_RESOURCE_GROUP = $ResourceGroup
    ACA_APP_NAME         = $acaName
}

foreach ($kv in $secrets.GetEnumerator()) {
    Write-Host "`n  Secret name : $($kv.Key)" -ForegroundColor Magenta
    Write-Host "  Value       :`n$($kv.Value)" -ForegroundColor White
}

Write-Host "`n  (Optional - used by the scheduled metrics workflow)" -ForegroundColor Gray
Write-Host "  Secret name : ACA_URL" -ForegroundColor Magenta
Write-Host "  Value       : https://$acaFqdn" -ForegroundColor White

Write-Host ("`n" + ("=" * 70)) -ForegroundColor Yellow

# -- 6. Summary ---------------------------------------------------------------
Write-Step "Setup complete!"
Write-Info "ACR login server  : $acrLoginServer"
Write-Info "ACA URL           : https://$acaFqdn"
Write-Info ""
Write-Info "Next steps:"
Write-Info "  1. Add the secrets above to GitHub"
Write-Info "  2. Push to main or trigger the workflow manually:"
Write-Info "     GitHub > Actions > '1 - Train -> Build -> Deploy to ACA' > Run workflow"
Write-Info ""
Write-Info "  First run trains the models (~10 min), builds the Docker image,"
Write-Info "  pushes it to ACR, and deploys it to ACA."