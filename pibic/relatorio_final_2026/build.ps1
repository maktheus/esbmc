param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Compilador pode ser TeX Live ou MiKTeX. O script não instala dependências.
$latex = Get-Command pdflatex -ErrorAction SilentlyContinue
$bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
if (-not $latex) {
    Write-Error 'pdflatex não foi encontrado no PATH. Instale TeX Live/MiKTeX ou abra este projeto no Overleaf.'
}

if ($Clean) {
    # Limpeza deliberadamente limitada à extensão dos artefatos auxiliares.
    $auxExtensions = @('.aux','.bbl','.bcf','.blg','.fdb_latexmk','.fls','.lof','.log','.lot','.out','.run.xml','.toc','.brf')
    Get-ChildItem -LiteralPath $PSScriptRoot -File | Where-Object { $auxExtensions -contains $_.Extension.ToLowerInvariant() } | Remove-Item -Force
}

& $latex.Source -interaction=nonstopmode -halt-on-error -file-line-error main.tex
if ($bibtex) {
    & $bibtex.Source main
}
& $latex.Source -interaction=nonstopmode -halt-on-error -file-line-error main.tex
& $latex.Source -interaction=nonstopmode -halt-on-error -file-line-error main.tex

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'main.pdf'))) {
    Write-Error 'A compilação terminou sem gerar main.pdf.'
}

$pdf = Get-Item -LiteralPath (Join-Path $PSScriptRoot 'main.pdf')
Write-Output ("PDF gerado: {0} bytes" -f $pdf.Length)

