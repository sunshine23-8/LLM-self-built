param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Main = "main"

if ($Clean) {
    Remove-Item -ErrorAction SilentlyContinue "$Main.aux", "$Main.bbl", "$Main.blg", "$Main.log", "$Main.out", "$Main.toc", "$Main.pdf"
    Write-Host "Cleaned LaTeX build artifacts."
    exit 0
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64",
        "C:\Program Files\MiKTeX\miktex\bin\x64"
    )
    foreach ($dir in $candidates) {
        if (Test-Path (Join-Path $dir "pdflatex.exe")) {
            $env:PATH = "$dir;$env:PATH"
            break
        }
    }
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Error "pdflatex not found. Please install MiKTeX (https://miktex.org/download) or TeX Live first."
}

pdflatex -interaction=nonstopmode "$Main.tex"
bibtex "$Main"
pdflatex -interaction=nonstopmode "$Main.tex"
pdflatex -interaction=nonstopmode "$Main.tex"
Write-Host "Built $Main.pdf successfully."
