#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Limpa arquivos PNG e DICOM duplicados do diretório archive.

.DESCRIPTION
    Remove todos os arquivos PNG e mantém apenas o primeiro arquivo DICOM
    (em ordem alfabética) em cada subdiretório do archive.

.PARAMETER BaseDir
    Diretório base a ser processado. Padrão: ./archive

.EXAMPLE
    .\clean_archive.ps1
    .\clean_archive.ps1 -BaseDir "./archive"
#>

param(
    [string]$BaseDir = "./archive"
)

# Verifica se o diretório existe
if (-not (Test-Path -Path $BaseDir -PathType Container)) {
    Write-Error "Diretório base '$BaseDir' não encontrado."
    exit 1
}

Write-Host "Processando diretório: $BaseDir" -ForegroundColor Cyan

# Obtém todos os subdiretórios recursivamente
$subdirs = Get-ChildItem -Path $BaseDir -Directory -Recurse

$totalDirs = $subdirs.Count
$currentDir = 0
$totalPngRemoved = 0
$totalDcmRemoved = 0

foreach ($dir in $subdirs) {
    $currentDir++
    Write-Host "`n[$currentDir/$totalDirs] Processando: $($dir.FullName)" -ForegroundColor Yellow
    
    # Remove todos os arquivos PNG
    $pngFiles = Get-ChildItem -Path $dir.FullName -Filter "*.png" -File -ErrorAction SilentlyContinue
    if ($pngFiles.Count -gt 0) {
        Write-Host "  Removendo $($pngFiles.Count) arquivo(s) PNG..." -ForegroundColor Gray
        $pngFiles | Remove-Item -Force -ErrorAction SilentlyContinue
        $totalPngRemoved += $pngFiles.Count
    }
    
    # Processa arquivos DICOM
    $dcmFiles = Get-ChildItem -Path $dir.FullName -Filter "*.dcm" -File -ErrorAction SilentlyContinue | Sort-Object Name
    if ($dcmFiles.Count -gt 1) {
        $keepFile = $dcmFiles[0]
        $removeFiles = $dcmFiles[1..($dcmFiles.Count - 1)]
        
        Write-Host "  Mantendo: $($keepFile.Name)" -ForegroundColor Green
        Write-Host "  Removendo $($removeFiles.Count) arquivo(s) DICOM duplicado(s)..." -ForegroundColor Gray
        
        foreach ($file in $removeFiles) {
            Write-Host "    - $($file.Name)" -ForegroundColor DarkGray
            Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
            $totalDcmRemoved++
        }
    } elseif ($dcmFiles.Count -eq 1) {
        Write-Host "  Apenas 1 arquivo DICOM encontrado (nada a remover)" -ForegroundColor Green
    } else {
        Write-Host "  Nenhum arquivo DICOM encontrado" -ForegroundColor DarkGray
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Limpeza concluida!" -ForegroundColor Green
Write-Host "Total de pastas processadas: $totalDirs" -ForegroundColor Cyan
Write-Host "Total de arquivos PNG removidos: $totalPngRemoved" -ForegroundColor Cyan
Write-Host "Total de arquivos DICOM removidos: $totalDcmRemoved" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

