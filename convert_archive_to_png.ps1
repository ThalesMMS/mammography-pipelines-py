#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Wrapper PowerShell para converter arquivos DICOM do archive para PNG.

.DESCRIPTION
    Chama o script Python convert_archive_to_png.py para converter todos os
    arquivos .dcm do diretório archive para PNG, mantendo a estrutura de pastas.

.PARAMETER SourceDir
    Diretório fonte com arquivos DICOM. Padrão: ./archive

.PARAMETER OutputDir
    Diretório de saída para PNGs. Padrão: ./archive_png

.PARAMETER SkipExisting
    Pula arquivos PNG que já existem.

.EXAMPLE
    .\convert_archive_to_png.ps1
    .\convert_archive_to_png.ps1 -SourceDir "./archive" -OutputDir "./archive_png"
    .\convert_archive_to_png.ps1 -SkipExisting
#>

param(
    [string]$SourceDir = "./archive",
    [string]$OutputDir = "./archive_png",
    [switch]$SkipExisting
)

$scriptPath = Join-Path $PSScriptRoot "convert_archive_to_png.py"

if (-not (Test-Path -Path $scriptPath)) {
    Write-Error "Script Python não encontrado: $scriptPath"
    exit 1
}

# Verifica se Python está disponível
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Usando Python: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Error "Python não encontrado. Certifique-se de que Python está instalado e no PATH."
    exit 1
}

# Monta os argumentos
$argsList = @(
    $scriptPath,
    "--source-dir", $SourceDir,
    "--output-dir", $OutputDir
)

if ($SkipExisting) {
    $argsList += "--skip-existing"
}

Write-Host "`nIniciando conversão DICOM para PNG..." -ForegroundColor Green
Write-Host "Diretório fonte: $SourceDir" -ForegroundColor Cyan
Write-Host "Diretório de saída: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Executa o script Python
python $argsList

if ($LASTEXITCODE -ne 0) {
    Write-Error "Erro ao executar o script Python. Código de saída: $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`nConversão concluída!" -ForegroundColor Green

