
$ErrorActionPreference = "Stop"

try 
{
  Push-Location $PSScriptRoot
  
  # Activate conda
  & $env:USERPROFILE\Anaconda3\shell\condabin\conda-hook.ps1 ; conda activate $env:USERPROFILE\Anaconda3
  
  # Check if the nyc-sc environment is installed
  $EnvName = "nyc-sc"
  $Environments = & conda env list
  $NycScEnvironment = $Environments | Where-Object { $_.Contains($EnvName) }

  if ($Null -eq $NycScEnvironment)
  {
    Write-Host "Installing $EnvName environment" -Foreground Cyan
    & conda env create -f nyc-sc.yml
  }
  else 
  {
    Write-Host "Updating $EnvName environment" -Foreground Cyan
    & conda activate nyc-sc
    & conda env update --file nyc-sc.yml
  }
}
finally
{
  # Conda has been activated, so now deactivate it
  if ($Null -eq (Get-Command conda))
  {
    & conda deactivate
  }
  Pop-Location
}

