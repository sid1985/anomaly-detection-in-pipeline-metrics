// Parameter file for infra/main.bicep
// =====================================
// Edit acrName to be globally unique before running setup.ps1
// The containerImage is overridden by the CI workflow at deploy time.

using './main.bicep'

param appName        = 'pl-paper6'
param acrName        = 'plpaper6acr'   // ACR names: alphanumeric only, no hyphens, globally unique
param logRetentionDays = 30
// containerImage is intentionally omitted here; CI passes it at deploy time.
